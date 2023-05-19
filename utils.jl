using MLDatasets

function normalize_input!(input::Array{Float32, 3}, mean::Float32, std::Float32)
    input[:] = (input .- mean) ./ std
end

function one_hot_target(labels::Vector, classes::Int64 = 10)
    output = zeros(Float32, classes, length(labels))

    for i in range(1,length(labels))
        output[labels[i]+1, i] = 1.0f0
    end

    return output
end

function MNIST_dataloader(path::String = "./benchmark/data/MNIST/raw", mean::Float32 = 0.1307f0, std::Float32 = 0.3081f0)
    train_x, train_y = MNIST(split=:train, dir=path)[:]
    test_x, test_y = MNIST(split=:test, dir=path)[:]

    normalize_input!(train_x, mean, std)
    normalize_input!(test_x, mean, std)
    train_y = one_hot_target(train_y)
    test_y = one_hot_target(test_y)
    
    return train_x, train_y, test_x, test_y
end

function step_lr(current_lr::Float32, epoch::Int64; gamma::Float32=0.9f0, step_size::Int64=1)
    if epoch % step_size == 0
        return current_lr*gamma
    end
        return current_lr
end    

function SGD(weights::Array{Float32}, gradient::Array{Float32}, lr::Float32)
    return weights .- gradient .* lr
end

function NLLLoss(y::Vector{Float32}, y_true::Vector{Float32})
    output, grad = 0.0f0, zeros(Float32, size(y_true))

    for i in 1:size(y)[1]
        if y_true[i] == 1.0
            output = -y[i]
            grad[i] = -1.0
            break
        end
    end
    return output, grad
end