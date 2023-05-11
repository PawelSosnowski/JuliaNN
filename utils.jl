using MLDatasets

function normalize_input!(input, mean, std)
    input[:] = (input .- mean) ./ std
end

function one_hot_target(labels::Vector)
    return Int32.(unique(labels) .== permutedims(labels))
end

function MNIST_dataloader(path::String = "./benchmark/data/MNIST/raw", mean::Float64 = 0.1307, std::Float64 = 0.3081)
    train_x, train_y = MNIST(split=:train, dir=path)[:]
    test_x, test_y = MNIST(split=:test, dir=path)[:]

    normalize_input!(train_x, mean, std)
    normalize_input!(test_x, mean, std)
    
    train_y = one_hot_target(train_y)
    test_y = one_hot_target(test_y)
    
    return train_x, train_y, test_x, test_y
end

function step_lr(current_lr, epoch; gamma=0.9, step_size=1)
    if epoch % step_size == 0
        return current_lr*gamma
    end
        return current_lr
end    

function SGD_step!(weights, gradient, lr)
    weights[:] = weights .- lr * gradient
end

