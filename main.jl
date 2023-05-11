using Random
include("nn.jl")
include("utils.jl")


if abspath(PROGRAM_FILE) == @__FILE__
    Random.seed!(123)
    println("Script start")

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

    model = NNGraph([
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

    train_x, train_y, test_x, test_y = MNIST_dataloader()

    for i in range(1,100)    
        x, y = train_x[:, :, i], train_y[:, i]
        y_hat = forward(reshape(x, (size(x)..., 1)), model)
        
        loss, loss_grad = NLLLoss(y_hat, y)[1]
        
        #TODO
        backward(loss_grad, model)
        # update_weights(model, optimizer) # optimizer is callable
        # zero_grad(model)
    end
end


