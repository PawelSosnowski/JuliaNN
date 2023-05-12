function conv2d_op!(output::Union{SubArray{Float32, 2}, Array{Float32, 2}}, input::Union{SubArray{Float32, 2}, Array{Float32, 2}}, filter::Union{SubArray{Float32, 2}, Array{Float32, 2}}, f_dim::Int64, out_dim::Int64)
    for i in 1:out_dim
        for j in 1:out_dim
            output[i, j] += sum(input[i:i+f_dim-1, j:j+f_dim-1] .* filter)
        end
    end
end

function conv2d_layer_op(input::Array{Float32, 3}, filters::Array{Float32, 4})
    f_dim, in_channels, out_channels = size(filters)[2:4]
    i_dim = size(input)[2]
    out_dim = i_dim - f_dim + 1

    output = zeros(Float32, out_dim, out_dim, out_channels)
    
    for n in 1:out_channels
        for c in 1:in_channels
            conv2d_op!(@view(output[:, :, n]), @view(input[:, :, c]), @view(filters[:,:, c, n]), f_dim, out_dim)
        end
    end

    return output
end

function conv2d_layer_grad_op(input::Array{Float32, 3}, filters::Array{Float32, 4}, grad::Array{Float32, 3})
    i_dim = size(input)[2]
    grad_dim = size(grad)[2]
    f_dim, in_channels, out_channels = size(filters)[2:4]
    
    padded_dim = size(input)[2] + f_dim - 1

    weights_grad = zeros(Float32, f_dim, f_dim, in_channels, out_channels)
    input_grad = zeros(Float32, i_dim, i_dim, in_channels)

    for n in out_channels
        padded_grad = zeros(Float32, padded_dim, padded_dim)
        padded_grad[f_dim: end - f_dim + 1, f_dim: end - f_dim + 1] = grad[:, :, n]

        for c in in_channels
            conv2d_op!(@view(input_grad[:, :, c]), padded_grad, reverse(filters[:, :, c, n]), f_dim, i_dim) # reverse(filters) (conv) padded(grad)
            conv2d_op!(@view(weights_grad[:, :, c, n]), @view(input[:, :, c]), @view(grad[:, :, n]), grad_dim, f_dim) # input (conv) grad
        end
    end
    
    return input_grad, weights_grad
end

function maxpool2d_op(input::Array{Float32, 3}, kernel_size::Int32)
    dim_i, n_filters = size(input)[2:3]
    out_dim = floor(Int, dim_i / kernel_size)
    output = Array{Float32, 3}(undef, out_dim, out_dim, n_filters)
    indices = zero(input)

    for n in 1:n_filters
        for i in 1:out_dim
            for j in 1:out_dim
                input_fragment = input[(i-1)*kernel_size+1:i*kernel_size, (j-1)*kernel_size+1:j*kernel_size, n]
                max_x, max_y = Tuple(argmax(input_fragment))
                max_x += (i-1)*kernel_size
                max_y += (j-1)*kernel_size
                indices[max_x, max_y, n] = 1
                output[i, j, n] = input[max_x, max_y, n]
            end
        end
    end
    return output, indices
end

function maxpool2d_grad_op(indices::Array{Float32, 3}, kernel_size::Int32, grad::Array{Float32, 3})
    input_grad = indices
    dim_i, n_filters = size(indices)[2:3]
    idx_range = floor(Int, dim_i / kernel_size)

    for n in 1:n_filters
        for i in 1:idx_range
            for j in 1:idx_range
                input_grad[(i-1)*kernel_size+1:i*kernel_size, (j-1)*kernel_size+1:j*kernel_size, n] .*= grad[i, j, n]
            end
        end
    end
    return input_grad
end