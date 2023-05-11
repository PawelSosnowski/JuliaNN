using Base

abstract type NNGraphNode end

abstract type NNLayer <: NNGraphNode end
abstract type NNOperator <: NNGraphNode end

Base.show(io::IO, layer::NNGraphNode) = print(io, typeof(layer),[(layer_param, getfield(layer, layer_param)) for layer_param in fieldnames(typeof(layer)) if !(typeof(getfield(layer, layer_param)) <: AbstractArray)]...)

struct NNGraph
    nodes::Vector{NNGraphNode}
end

Base.iterate(graph::NNGraph) = Base.iterate(graph.nodes)
Base.iterate(graph::NNGraph, state::Int64) = Base.iterate(graph.nodes, state)
Base.getindex(graph::NNGraph, index::Int64) = Base.getindex(graph.nodes, index)
Base.getindex(graph::NNGraph, r::UnitRange{Int64}) = Base.getindex(graph.nodes, r) 
Base.lastindex(graph::NNGraph) = Base.lastindex(graph.nodes)

function showNNGraph(io::IO, graph::NNGraph)
    print(io, "Computational graph:\n(\n")
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

function backward(graph::NNGraph)

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
    return x
end

function backward(x::Array{Float32}, layer::Convolution2D)

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
    return x
end

function backward(x::Array{Float32}, layer::Linear)

end

mutable struct RELU <: NNOperator
    input_shape::Tuple{Int32, Int32, Int32}
    input::Array{Float32, 3}
    input_grad::Array{Float32, 3}

    RELU(inshape_) = new(inshape_, Array{Float32, 3}(undef, inshape_...), zeros(inshape_...))
end

function forward(x::Array{Float32}, operator::RELU)
    return x
end

function backward(x::Array{Float32}, operator::RELU)

end

mutable struct MaxPool2D <: NNOperator
    kernel_size::Int32

    input_shape::Tuple{Int32, Int32, Int32}
    input::Array{Float32, 3}
    input_grad::Array{Float32, 3}

    chosen_indices::Array{Float32, 3}

    MaxPool2D(kernel_, inshape_) = new(kernel_, inshape_, Array{Float32, 3}(undef, inshape_...), zeros(inshape_...), zeros(inshape_...))
end

function forward(x::Array{Float32}, operator::MaxPool2D)
    return x
end

function backward(x::Array{Float32}, operator::MaxPool2D)

end

mutable struct Flatten <: NNOperator
    input_shape::Tuple{Int32, Int32, Int32}
    input::Array{Float32, 3}
    input_grad::Array{Float32, 3}

    Flatten(inshape_) = new(inshape_, Array{Float32, 3}(undef, inshape_...), zeros(inshape_...))
end

function forward(x::Array{Float32}, operator::Flatten)
    return x
end

function backward(x::Array{Float32}, operator::Flatten)

end

mutable struct LogSoftmax <: NNOperator
    input_shape::Tuple{Int32, Int32}
    input::Array{Float32, 2}
    input_grad::Array{Float32, 2}

    LogSoftmax(inshape_) = new(inshape_, Array{Float32, 2}(undef, inshape_...), zeros(inshape_...))
end

function forward(x::Array{Float32}, operator::LogSoftmax)
    return x
end

function backward(x::Array{Float32}, operator::LogSoftmax)

end

function uniform_init_weights(neurons, weights_shape)
    stdv = 1. / sqrt(neurons)
    return 2 .* stdv .* rand(weights_shape...) .- stdv
end


# loss