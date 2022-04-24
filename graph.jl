abstract type GraphNode end
abstract type GraphOperator <: GraphNode end

struct Constant{T} <: GraphNode
    outputs:: T
    name :: String
    Constant(value; name="?") = new{typeof(value)}(value, name)
end

mutable struct Variable <: GraphNode
    outputs :: Any
    gradient :: Any
    name :: String
    Variable(value; name="?") = new(value, nothing, name)
end

mutable struct ScalarOperator{F} <: GraphOperator
    inputs :: Any
    outputs :: Any
    gradient :: Any
    name :: String
    ScalarOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end


mutable struct VectorOperator{F} <: GraphOperator
    inputs :: Any
    outputs :: Any
    gradient :: Any
    name :: String
    VectorOperator(fun, inputs...; name="?") = new{typeof(fun)}(inputs, nothing, nothing, name)
end

# Graph visual representation
import Base: show, summary
show(io::IO, operator::ScalarOperator{F}) where {F} = begin
    print(io, "Operator ", operator.name, " (", F, ")")
    print(io, "\n    ⏬")
end
show(io::IO, operator::VectorOperator{F}) where {F} = begin
    print(io, "Operator. ", operator.name, " (", F, ")")
    print(io, "\n    ⏬")
end
show(io::IO, constant::Constant) = begin
    print(io, "Constant ", constant.name)
    print(io, "\n  value: "); summary(io, constant.outputs)
    print(io, "\n    ⏬")
end
show(io::IO, var::Variable) = begin 
    print(io, "Variable ", var.name)
    print(io,"\n  value: "); summary(io, var.outputs)
    print(io,"\n  gradient: "); summary(io, var.gradient)
    print(io, "\n    ⏬")
end

# Graph building

function create_graph(head::GraphNode)
    visited = Set()
    ordered_graph = Vector()
    visit(head, visited, ordered_graph)
    return ordered_graph
end

function visit(node::GraphOperator, visited, order)
    if node ∉ visited
        push!(visited, node)
        for input in node.inputs
            visit(input, visited, order)
        end
        push!(order, node)
    end
    return nothing
end

function visit(node::GraphNode, visited, order)
    if node ∉ visited
        push!(visited, node)
        push!(order, node)
    end
    return nothing
end

# Forward graph pass

function forward!(graph::Vector)
    for node in graph
        if typeof(node) <: GraphOperator
            node.outputs = forward(node, [input.outputs for input in node.inputs]...)
        end
        zero_grad!(node)
    end
    return last(graph).outputs
end

zero_grad!(node::Constant) = nothing
zero_grad!(node::Variable) = node.gradient = nothing
zero_grad!(node::GraphOperator) = node.gradient = nothing

# Backward graph pass

function backward!(node::Constant) end
function backward!(node::Variable) end
function backward!(node::GraphOperator)
    inputs = node.inputs
    gradients = backward(node, [input.outputs for input in inputs]..., node.gradient)
    for (input, gradient) in zip(inputs, gradients)
        update_grad!(input, gradient)
    end
    return nothing
end    

function backward!(graph::Vector; seed=1.0)
    result = last(graph)
    result.gradient = seed
    #TODO: Define gradient for vector functions
    @assert length(result.outputs) == 1 
    for node in reverse(graph)
        backward!(node)
    end
    return nothing
end

update_grad!(node::Constant, gradient) = nothing
update_grad!(node::GraphNode, gradient) = let
    if isnothing(node.gradient)
        node.gradient = gradient 
    else 
        node.gradient .+= gradient
    end
end

# Operations supported in graph

import Base: ^
^(x::GraphNode, n::GraphNode) = ScalarOperator(^, x, n)
forward(::ScalarOperator{typeof(^)}, x, n) = return x^n
backward(::ScalarOperator{typeof(^)}, x, n, g) = tuple(g * n * x ^ (n-1), g * log(abs(x)) * x ^ n)

import Base: *
import LinearAlgebra: mul!

# matrix multiplication
*(x::GraphNode, y::GraphNode) = VectorOperator(mul!, x, y)
forward(::VectorOperator{typeof(mul!)}, x, y) = return x * y
backward(::VectorOperator{typeof(mul!)}, x, y, g) = tuple(g * y', x' * g)

# x .* y element-wise multiplication
Base.Broadcast.broadcasted(*, x::GraphNode, y::GraphNode) = VectorOperator(*, x, y)
forward(::VectorOperator{typeof(*)}, x, y) = return x .* y
backward(node::VectorOperator{typeof(*)}, x, y, g) = let
    I = ones(length(node.outputs))
    Jx = diagm(vec(y .* I))
    Jy = diagm(vec(x .* I))
    tuple(Jx' * g, Jy' * g)
end

# x ./ y element-wise multiplication
Base.Broadcast.broadcasted(/, x::GraphNode, y::GraphNode) = VectorOperator(/, x, y)
forward(::VectorOperator{typeof(/)}, x, y) = return x ./ y
backward(node::VectorOperator{typeof(/)}, x, y, g) = let
    I = ones(length(node.outputs))
    Jx = diagm(vec(I ./ y))
    Jy = (-x ./ y .^2)
    tuple(Jx' * g, Jy' * g)
end

Base.Broadcast.broadcasted(-, x::GraphNode, y::GraphNode) = VectorOperator(-, x, y)
forward(::VectorOperator{typeof(-)}, x, y) = return x .- y
backward(::VectorOperator{typeof(-)}, x, y, g) = tuple(g, -g)

Base.Broadcast.broadcasted(+, x::GraphNode, y::GraphNode) = VectorOperator(+, x, y)
forward(::VectorOperator{typeof(+)}, x, y) = return x .+ y
backward(::VectorOperator{typeof(+)}, x, y, g) = tuple(g, g)

Base.Broadcast.broadcasted(exp, x::GraphNode) = VectorOperator(exp, x)
forward(::VectorOperator{typeof(exp)}, x) = return exp.(x)
backward(::VectorOperator{typeof(exp)}, x, g) = let
    return tuple(g .* exp.(x))
end

import Base: sum
sum(x::GraphNode) = VectorOperator(sum, x)
forward(::VectorOperator{typeof(sum)}, x) = return sum(x)
backward(::VectorOperator{typeof(sum)}, x, g) = let
    I = ones(length(x))
    J = I'
    tuple(J' * g)
end

import Base: max
Base.Broadcast.broadcasted(max, x::GraphNode, y::GraphNode) = VectorOperator(max, x, y)
forward(::VectorOperator{typeof(max)}, x, y) = return max.(x, y)
backward(::VectorOperator{typeof(max)}, x, y, g) = let
    Jx = diagm(isless.(y, x))
    Jy = diagm(isless.(x, y))
    tuple(Jx' * g, Jy' * g)
end

import Base: one, zero
Base.Broadcast.broadcasted(one, x::GraphNode) = VectorOperator(one, x)
forward(::VectorOperator{typeof(one)}, x) = return one.(x)
backward(::VectorOperator{typeof(one)}, x, g) = return tuple(g .* one.(x))

Base.Broadcast.broadcasted(zero, x::GraphNode) = VectorOperator(zero, x)
forward(::VectorOperator{typeof(zero)}, x) = return zero.(x)
backward(::VectorOperator{typeof(zero)}, x, g) = return tuple(g .* zero.(x))
