using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy, throttle, @epochs, crossentropy #Note on crossentropy,
#when using softmax layer use crossentropy otherwise use logitcrossentropy
using Base.Iterators: repeated
using Parameters: @with_kw
using CUDAapi
using Printf, BSON
using CSV
using DelimitedFiles
if has_cuda()
    @info "CUDA is on"
    import CuArrays
    CuArrays.allowscalar(false)
end

@with_kw mutable struct Args
    η::Float64 = 1e-3
    epochs::Int = 50
    batch_size::Int = 128
    savepath::String = "./"
    device::Function = gpu
end


function getdata(args)
    # Loading Dataset
    df=CSV.read("/home/jawla/Simulations/DNN_gonihedric_data.csv")
    len=size(df, 1)
    split_boundary=ceil(Int64,0.9*len)
    df=df[shuffle(1:len),:]
    train=df[1:split_boundary,:]
    validate=df[split_boundary+1:len,:]
    x_train=permutedims(Matrix(select(train, Not([:phase,:temperature,:energy,:autocorrelation]))))
    y_train=permutedims(Matrix(select(train,:phase)))
    # One-hot-encode the labels
    y_train= onehotbatch(y_train, 0:1)#make onehotbatch of lables, essentially columns of labels
    # Batching
    train_data = DataLoader(x_train, y_train, batch_size=args.batch_size)#makes an array of tuples(batches) where each batch is a matrix with columsn as samples, and rows as features

    x_validate=permutedims(Matrix(select(validate, Not([:phase,:temperature,:energy,:autocorrelation]))))
    y_validate=permutedims(Matrix(select(validate,:phase)))
    y_validate= onehotbatch(validate, 0:1)
    validate_data = DataLoader(x_validate, y_validate, batch_size=args.batch_size)#
    return train_data,validate_data
end

function get_test_data(args)
    # Loading Dataset
    df=CSV.read("/home/jawla/jobs/DNN_gonihedric_test.csv")
    x=permutedims(Matrix(select(df, Not([:phase,:temperature,:energy,:autocorrelation]))))
    temps=permutedims(Matrix(select(df,:temperature)))
    return x,temps
end

function build_model(args; n_features=518, nclasses=2)
    return Chain(Dense(n_features, 128, relu),Dense(128, 64, relu),BatchNorm(64, relu),Dense(64, 32, relu), Dense(32, nclasses),softmax)
end

# Returns a vector of all parameters used in model
paramvec(m) = vcat(map(p->reshape(p, :), params(m))...)

# Function to check if any element is NaN or not
anynan(x) = any(isnan.(x))

function accuracy(data_loader, model)
    acc = 0
    for (x,y) in data_loader
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
    end
    acc/length(data_loader)
end

function train(; kws...)
    # Initializing Model parameters
    args = Args(; kws...)

    @info("Loading data set")
    train_set,validate_set= getdata(args)

    # Construct model
    @info("Building model...")
    model = build_model(args)
    # Load model and datasets onto GPU, if enabled
    train_set = args.device.(train_set)
    validate_set = args.device.(validate_set)
    model = args.device(model)

    # Make sure our model is nicely precompiled before starting our training loop
    model(train_set[1][1])

    loss(x,y) = crossentropy(model(x), y)

    # Train our model with the given training set using the ADAM optimizer
    opt = ADAM(args.η)
    @info("Beginning training loop...")
    best_acc = 0.0
    last_improvement = 0
    for epoch_idx in 1:args.epochs
        # Train for a single epoch
        Flux.train!(loss, params(model), train_set, opt)

        # Terminate on NaN
        if anynan(paramvec(model))
            @error "NaN params"
            break
        end
        # Calculate accuracy:
        acc = accuracy(validation_set, model)

        @info(@sprintf("[%d]: Test accuracy: %.4f", epoch_idx, acc))
        # If our accuracy is good enough, quit out.
        if acc >= 0.999
            @info(" -> Early-exiting: We reached our target accuracy of 99.9%")
            break
        end

        # If this is the best accuracy we've seen so far, save the model out
        if acc >= best_acc
            @info(" -> New best accuracy! Saving model out to mlp_gonihedric.bson")
            BSON.@save joinpath(args.savepath, "mlp_gonihedric.bson") params=cpu.(params(model)) epoch_idx acc
            best_acc = acc
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
    test_set,temp_list = get_test_data(args)
    # Re-constructing the model with random initial weights
    model = build_model(args)
    # Loading the saved parameters
    BSON.@load joinpath(args.savepath, "mlp_gonihedric.bson") params
    # Loading parameters onto the model
    Flux.loadparams!(model, params)
    prob_dist=model(test_set)
    results=vcat(prob_dist,temp_list)
    writedlm("DNN_gonihedric_results.csv",results)
    # activations(model, input for model=rand(10)) ; sum(vecnorm, ans) #check activations and some othe runknow function
end

cd(@__DIR__)
train()
test()
