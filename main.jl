using Random

include("nn.jl")
include("utils.jl")


function train(train_x::Array{Float32, 3}, train_y::Matrix{Float32}, model::NNGraph, learning_rate::Float32)
    for i in range(1, size(train_y)[2])
        x, y = train_x[:, :, i], train_y[:, i]

        y_hat = forward(reshape(x, (size(x)..., 1)), model)
        
        y_hat = reshape(y_hat, size(y_hat)[1])
        loss, loss_grad = NLLLoss(y_hat, y)

        if i % (size(train_y)[2] / 1000) == 0
            println("[", i / size(train_y)[2] * 100 ,"%]", "Loss: ", loss)
        end
        
        backward(loss_grad, model)
        update_weights(model, SGD, learning_rate)
        zero_grad(model)
    end
end

function evaluate(test_x::Array{Float32, 3}, test_y::Matrix{Float32}, model::NNGraph) 
    loss = 0.0f0
    for i in range(1, size(test_y)[2])
        x, y = test_x[:, :, i], test_y[:, i]

        y_hat = forward(reshape(x, (size(x)..., 1)), model)
        loss += NLLLoss(y_hat, y)[1]
    end
    println("Average test loss: ", loss / size(test_y)[2])
end

function main()
    Random.seed!(1) # for reproducility
    learning_rate = 0.001f0
    epochs = 1

    model_dims = [
        (28, 28, 1), 
        (26, 26, 32), 
        (26, 26, 32), 
        (13, 13, 32), 
        (11, 11, 64), 
        (11, 11, 64), 
        (5, 5, 64), 
        (1600, 1), 
        (10, 1)
    ]

    model = NNGraph("CNNet", [
        Convolution2D(28, 3, 1, 32, model_dims[1]),
        RELU(model_dims[2]),
        MaxPool2D(2, model_dims[3]),
        Convolution2D(13, 3, 32, 64, model_dims[4]),
        RELU(model_dims[5]),
        MaxPool2D(2, model_dims[6]),
        Flatten(model_dims[7]),
        Linear(1600, 10, model_dims[8]),
        LogSoftmax(model_dims[9])
    ])

    println(model)

    train_x, train_y, test_x, test_y = MNIST_dataloader()

    for epoch in range(1, epochs)
        train(train_x, train_y, model, learning_rate)
        evaluate(test_x, test_y, model)
        learning_rate = step_lr(learning_rate, epoch)
    end

end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end


# time test:
# using @benchmark train($train_x, $train_y, $model, $lr) - mean: 46.08 min

# memory test:
# using @benchamrk 
# - Model: 2.56MB, allocs: 133
# - Model with train/test data in memory: 499.92MB, allocs: 346
# - Model with train/test data in memory and 1 train loop: 554.005MB, allocs: 555278