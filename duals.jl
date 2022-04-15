import Base: +, -, *, /, abs, sin, cos, tan, exp, sqrt, isless

# Dual number definition: Dual z = a + bϵ, where a, b ∈ ℝ and ϵ^2 = 0
struct Dual{T <: Real} <: Number
    v::T
    dv::T
end

function Dual(v :: T) where T <: Real
    return Dual(v, one(v))
end

# implement basic operators for duals
-(x::Dual) = Dual(-x.v, -x.dv)
+(x::Dual, y::Dual) = Dual(x.v + y.v, x.dv + y.dv)
-(x::Dual, y::Dual) = Dual(x.v - y.v, x.dv - y.dv)
*(x::Dual, y::Dual) = Dual(x.v*y.v, x.dv*y.v + x.v*y.dv)
/(x::Dual, y::Dual) = Dual(x.v / y.v, (x.dv*y.v + x.v*y.dv)/y.v^2)

isless(x::Dual, y::Dual) = x.v < y.v
abs(x::Dual) = Dual(abs(x.v), sign(x.v)*x.dv)
sin(x::Dual) = Dual(sin(x.v), cos(x.v)*x.dv)
cos(x::Dual) = Dual(cos(x.v), -sin(x.v)*x.dv)
tan(x::Dual) = Dual(tan(x.v), one(x.v)*x.dv + tan(x.v)^2*x.dv)
exp(x::Dual) = Dual(exp(x.v), exp(x.v)*x.dv)
sqrt(x::Dual) = Dual(sqrt(x.v), .5/sqrt(x.v)*x.dv)

import Base: show

show(io::IO, x::Dual) = print(io, '[', x.v, ", ", x.dv, "ϵ]")
value(x::Dual) = x.v
derivative(x::Dual) = x.dv

import Base: promote_rule, convert
convert(::Type{Dual{T}}, x::Dual) where T = Dual(convert(T, x.v), convert(T, x.dv))
convert(::Type{Dual{T}}, x::Number) where T = Dual(convert(T, x), zero(T))
promote_rule(::Type{Dual{T}}, ::Type{R}) where {T,R} = Dual{promote_type(T,R)}


ϵ = Dual(0., 1.);