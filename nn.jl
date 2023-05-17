using Base
include("backend.jl")

abstract type NNGraphNode end

abstract type NNLayer <: NNGraphNode end

function zero_grad(layer::NNLayer)
    layer.input_grad .*= 0.0f0
    layer.weights_grad .*= 0.0f0

    if :bias_grad in fieldnames(typeof(layer))
        layer.bias_grad .*= 0.0f0
    end
    layer.input .*= 0.0f0
end

abstract type NNOperator <: NNGraphNode end

function zero_grad(operator::NNOperator)
    operator.input_grad .*= 0.0f0
    operator.input .*= 0.0f0
end

Base.show(io::IO, layer::NNGraphNode) = print(io, typeof(layer),[(layer_param, getfield(layer, layer_param)) for layer_param in fieldnames(typeof(layer)) if !(typeof(getfield(layer, layer_param)) <: AbstractArray)]...)

struct NNGraph
    name::String
    nodes::Vector{NNGraphNode}
    # Feature to add: inference_mode::Bool that indicates whether to store data for backprop in graph.
end

Base.iterate(graph::NNGraph) = Base.iterate(graph.nodes)
Base.iterate(graph::NNGraph, state::Int64) = Base.iterate(graph.nodes, state)
Base.getindex(graph::NNGraph, index::Int64) = Base.getindex(graph.nodes, index)
Base.getindex(graph::NNGraph, r::UnitRange{Int64}) = Base.getindex(graph.nodes, r) 
Base.lastindex(graph::NNGraph) = Base.lastindex(graph.nodes)
Base.reverse(graph::NNGraph) = Base.reverse(graph.nodes)

function showNNGraph(io::IO, graph::NNGraph)
    print(io, graph.name, ":\n(\n")
    print(io, graph[1])

    for node in graph[2:end]
        print(io, '\n')
        print(io, "         ↓         \n")
        print(io, node)
    end
    print(io, "\n)")
end
Base.show(io::IO, graph::NNGraph) = showNNGraph(io, graph)


function forward(data::Array{Float32, 3}, graph::NNGraph)
    for node in graph
        data = forward(data, node)
    end
    return data
end

function backward(gradient::Vector{Float32}, graph::NNGraph)
    for node in reverse(graph)
        backward(gradient, node)
        gradient = node.input_grad
    end
end

function zero_grad(graph::NNGraph)
    for node in graph
        zero_grad(node)
    end
end

function update_weights(graph::NNGraph, optimizer::Function, lr::Float32)
    for node in graph
        if typeof(node) <: NNLayer
            update_weights(node, optimizer, lr)
        end
    end
end

mutable struct Convolution2D <: NNLayer
    input_size::Int32
    kernel_size::Int32

    in_filters::Int32
    out_filters::Int32

    input_shape::Tuple{Int32, Int32, Int32}

    input::Array{Float32, 3}
    weights::Array{Float32, 4}

    input_grad::Array{Float32, 3}
    weights_grad::Array{Float32, 4}

    Convolution2D(in_,  kernel_, inf_, outf_, insh_) = new(in_, kernel_, inf_, outf_, insh_,
                                                           Array{Float32, 3}(undef, insh_...),
                                                           uniform_init_weights(inf_, (kernel_, kernel_, inf_, outf_)),
                                                           zeros(insh_...), 
                                                           zeros(kernel_, kernel_, inf_, outf_))
end

function forward(x::Array{Float32}, layer::Convolution2D)
    layer.input = x
    return conv2d_layer_op(layer.input, layer.weights)
end

function backward(gradient::Array{Float32}, layer::Convolution2D)
    layer.input_grad, layer.weights_grad = conv2d_layer_grad_op(layer.input, layer.weights, gradient)
end

function update_weights(layer::Convolution2D, optimizer::Function, lr::Float32)
    layer.weights = optimizer(layer.weights, layer.weights_grad, lr)
end

mutable struct Linear <: NNLayer
    input_neurons::Int32
    output_neurons::Int32

    input_shape::Tuple{Int32, Int32}

    input::Array{Float32, 2}
    weights::Array{Float32, 2}
    bias::Array{Float32, 1}

    input_grad::Array{Float32, 2}
    weights_grad::Array{Float32, 2}
    bias_grad::Array{Float32, 1}

    Linear(inn_, outn_, insh_) = new(inn_, outn_, insh_,
                                     Array{Float32, 2}(undef, insh_...),
                                     uniform_init_weights(inn_, (outn_, inn_)),
                                     uniform_init_weights(inn_, outn_),
                                     zeros(insh_...), 
                                     zeros(outn_, inn_),
                                     zeros(outn_))

end

function forward(x::Array{Float32}, layer::Linear)
    layer.input = x
    return layer.weights * layer.input + layer.bias
end

function backward(gradient::Array{Float32}, layer::Linear)
    layer.weights_grad = gradient * layer.input'
    layer.bias_grad = reshape(gradient, size(gradient)[1])
    layer.input_grad = layer.weights' * gradient
end

function update_weights(layer::Linear, optimizer::Function, lr::Float32)
    layer.weights = optimizer(layer.weights, layer.weights_grad, lr)
    layer.bias = optimizer(layer.bias, layer.bias_grad, lr)
end

mutable struct RELU <: NNOperator
    input_shape::Tuple{Int32, Int32, Int32}
    input::Array{Float32, 3}
    input_grad::Array{Float32, 3}

    RELU(inshape_) = new(inshape_, Array{Float32, 3}(undef, inshape_...), zeros(inshape_...))
end

function forward(x::Array{Float32}, operator::RELU)
    operator.input = x
    return max.(0, operator.input)
end

function backward(gradient::Array{Float32}, operator::RELU)
    operator.input_grad = gradient
    operator.input_grad[operator.input .<= 0.0f0] .= 0.0f0
end

mutable struct MaxPool2D <: NNOperator
    kernel_size::Int32

    input_shape::Tuple{Int32, Int32, Int32}
    input::Array{Float32, 3}
    input_grad::Array{Float32, 3}

    max_indices::Array{Float32, 3}

    MaxPool2D(kernel_, inshape_) = new(kernel_, inshape_, Array{Float32, 3}(undef, inshape_...), zeros(inshape_...), zeros(inshape_...))
end

function forward(x::Array{Float32}, operator::MaxPool2D)
    operator.input = x
    transformed_x, operator.max_indices = maxpool2d_op(operator.input, operator.kernel_size)
    return transformed_x
end

function backward(gradient::Array{Float32}, operator::MaxPool2D)
    operator.input_grad = maxpool2d_grad_op(operator.max_indices, operator.kernel_size, gradient)
end

mutable struct Flatten <: NNOperator
    input_shape::Tuple{Int32, Int32, Int32}
    input::Array{Float32, 3}
    input_grad::Array{Float32, 3}

    Flatten(inshape_) = new(inshape_, Array{Float32, 3}(undef, inshape_...), zeros(inshape_...))
end

function forward(x::Array{Float32}, operator::Flatten)
    operator.input = x
    return reshape(operator.input, prod(size(operator.input)), 1)
end

function backward(gradient::Array{Float32}, operator::Flatten)
    operator.input_grad = reshape(gradient, operator.input_shape)
end

mutable struct LogSoftmax <: NNOperator
    input_shape::Tuple{Int32, Int32}
    input::Array{Float32, 2}
    input_grad::Array{Float32, 2}

    LogSoftmax(inshape_) = new(inshape_, Array{Float32, 2}(undef, inshape_...), zeros(inshape_...))
end

function forward(x::Array{Float32}, operator::LogSoftmax)
    operator.input = x
    c = maximum(operator.input)
    return operator.input .- (c + log(sum(exp.(operator.input .- c))))
end

function backward(gradient::Array{Float32}, operator::LogSoftmax)
    c = maximum(operator.input)
    operator.input_grad = exp.(operator.input .- (c + log(sum(exp.(operator.input .- c))))) .+ gradient
end

function uniform_init_weights(neurons, weights_shape)
    stdv = 1. / sqrt(neurons)
    return 2 .* stdv .* rand(weights_shape...) .- stdv
end