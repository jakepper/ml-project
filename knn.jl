using CSV, Tables, Statistics, Distributed

# This line adds 2 parallel processes to the program
# increasing the speed by a factor of approximately 2.     
# You can choose a different number if you have more 
# available cores in your machine.
addprocs(3) 

@everywhere function euclidean_distance(a, b)
    distance = 0.0
    for index in eachindex(a)
        distance += (a[index] - b[index]) ^ 2
    end
    return sqrt(distance)
end

@everywhere function get_knn(x, i, k)
    nRows, nCols = size(x)

    imageJ = Array{Float32}(undef, nCols)
    
    distances = Array{Float32}(undef, nRows)
    
    for index in 1:nRows
        # The next for loop fills the vector image_j with the j data point 
        # from the main matrix. Copying element one by one is faster
        # than copying the entire vector at once.
        for j in 1:nCols
            imageJ[j] = x[index, j]
        end
        # calculate the distance and save the result
        distances[index] = euclidean_distance(i, imageJ)
    end
    
    # indices sorted by distances.
    sortedNeighbors = sortperm(distances)
    
    # select the k nearest neighbors
    kNearestNeighbors = sortedNeighbors[1:k]
    return kNearestNeighbors
end

@everywhere function assign_label(x, y, k, i)
    kNearestNeighbors = get_knn(x, i, k)
    
    # let's make a dictionary to save the counts of 
    # the labels
    # Dict{}() is also right .
    # Int,Int indicates the dictionary to expect integer values 
    counts = Dict{Int, Int}() 

    # The next two variables keep track of the 
    # label with the highest count.
    highestCount = 0
    mostPopularLabel = 0

    # Iterating over the labels of the k nearest neighbors
    for n in kNearestNeighbors
        labelOfN = y[n]

        # Adding the current label to our dictionary
        # if it's not already there
        if !haskey(counts, labelOfN)
            counts[labelOfN] = 0
        end

        # Add one to the count
        counts[labelOfN] += 1 

        if counts[labelOfN] > highestCount
            highestCount = counts[labelOfN]
            mostPopularLabel = labelOfN
        end 
    end
    return mostPopularLabel
end

# Read training matrix
xTrain = CSV.File("./xTrain.csv") |> Tables.matrix

# Read test matrix
xTest = CSV.File("./xTest.csv") |> Tables.matrix

# Read yTrain
yTrain = CSV.File("./yTrain.csv") |> Tables.matrix
yTrain = vec(yTrain)

# Read information about test data ( IDs ).
labelsInfoTest = DataFrame(CSV.File("./sampleSubmission.csv"))

nRows = size(xTest, 1)
k = 5 # Best K
pred = @distributed (vcat) for i in 1:nRows
    nCols = size(xTest, 2)
    imageI = Array{Float32}(undef, nCols)
    for j in 1:nCols
        imageI[j] = xTest[i, j]
    end
    assign_label(xTrain, yTrain, k, imageI)
end

#Convert integer predictions to character
labelsInfoTest[:,"Class"] = string.(Char.(pred))

#Save predictions
CSV.write("./juliaKNNSubmission.csv", labelsInfoTest)