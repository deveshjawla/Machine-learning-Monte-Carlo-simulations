using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs, crossentropy #Note on crossentropy,
#when using softmax layer use crossentropy otherwise use logitcrossentropy
using Base.Iterators: repeated
using Parameters: @with_kw
using CUDAapi
using Printf, BSON
using CSV
if has_cuda()
    @info "CUDA is on"
    import CuArrays
    CuArrays.allowscalar(false)
end

@with_kw mutable struct Args
    η::Float64 = 3e-2
    epochs::Int = 50
    batch_size::Int = 128
    savepath::String = "./"
    device::Function = gpu
end


function getdata(args)
    # Loading Dataset
    df=CSV.read("/home/jawla/jobs/DNN_gonihedric_data.csv")
    x=permutedims(Matrix(select(df, Not([:T,:phase]))))
    y=permutedims(Matrix(select(df,:phase)))
    # One-hot-encode the labels
    y= onehotbatch(y, 0:1)#make onehotbatch of lables, essentially columns of labels
    # Batching
    train_data = DataLoader(x, y, batch_size=args.batch_size, shuffle=true)#makes an array of tuples(batches) where each batch is a matrix with columsn as samples, and rows as features
    return train_data
end

function get_test_data(args)
    # Loading Dataset
    df=CSV.read("/home/jawla/jobs/DNN_gonihedric_data_Tc.csv")
    x=permutedims(Matrix(select(df, Not([:T,:phase]))))
    y=permutedims(Matrix(select(df,:phase)))
    # One-hot-encode the labels
    y= onehotbatch(y, 0:1)#make onehotbatch of lables, essentially columns of labels
    # Batching
    test_data = DataLoader(x, y, batch_size=args.batch_size, shuffle=true)#makes an array of tuples(batches) where each batch is a matrix with columsn as samples, and rows as features
    return test_data
end

function build_model(args; n_features=92, nclasses=2)
    return Chain(Dense(n_features, 128, relu),Dense(128, 64, relu),BatchNorm(64, relu),Dense(64, 32, relu), Dense(32, nclasses),softmax)
end

# Returns a vector of all parameters used in model
paramvec(m) = vcat(map(p->reshape(p, :), params(m))...)

# Function to check if any element is NaN or not
anynan(x) = any(isnan.(x))

function loss_all(dataloader, model)
    l = 0f0
    for (x,y) in dataloader
        l += logitcrossentropy(model(x), y)
    end
    l/length(dataloader)
end

function train(; kws...)
    # Initializing Model parameters
    args = Args(; kws...)

    @info("Loading data set")
    train_data= getdata(args)

    # Construct model
    @info("Building model...")
    model = build_model(args)
    # Load model and datasets onto GPU, if enabled
    train_data = args.device.(train_data)
    model = args.device(model)

    # Make sure our model is nicely precompiled before starting our training loop
    model(train_data[1][1])

    loss(x,y) = crossentropy(model(x), y)

    # Train our model with the given training set using the ADAM optimizer and
    # printing out performance against the test set as we go.
    evalcb = () -> @show(loss(X, Y))
    opt = ADAM(args.η)
    @info("Beginning training loop...")
        min_loss = 0.0
        last_improvement = 0
        for epoch_idx in 1:args.epochs
            # Train for a single epoch
            Flux.train!(loss, params(model), train_data, opt,cb = throttle(evalcb, 10))

            # Terminate on NaN
            if anynan(paramvec(model))
                @error "NaN params"
                break
            end
            loss_=loss_all(train_data, model)
            # If this is the best accuracy we've seen so far, save the model out
            if  loss_<= min_loss
                @info(" -> New best minimum loss! Saving model out to mlp_gonihedric.bson")
                BSON.@save joinpath(args.savepath, "mlp_gonihedric.bson") params=cpu.(params(model)) epoch_idx acc
                min_loss = acc
                last_improvement = epoch_idx
            end

            # If we haven't seen improvement in 5 epochs, drop our learning rate:
            if epoch_idx - last_improvement >= 5 && opt.eta > 1e-6
                opt.eta /= 10.0
                @warn(" -> Haven't improved in a while, dropping learning rate to $(opt.eta)!")

                # After dropping learning rate, give it a few epochs to improve
                last_improvement = epoch_idx
            end

            if epoch_idx - last_improvement >= 10
                @warn(" -> We're calling this converged.")
                break
            end
        end
end

function test(; kws...)
    args = Args(; kws...)

    # Loading the test data
    test_set = get_test_data(args)

    # Re-constructing the model with random initial weights
    model = build_model(args)

    # Loading the saved parameters
    BSON.@load joinpath(args.savepath, "mlp_gonihedric.bson") params

    # Loading parameters onto the model
    Flux.loadparams!(model, params)

    test_set = gpu.(test_set)
    model = gpu(model)
 # activations(model, input for model=rand(10)) ; sum(vecnorm, ans) #check activations and some othe runknow function

end

cd(@__DIR__)
train()
test()
