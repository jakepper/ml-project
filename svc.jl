using CSV, DataFrames, Tables, Statistics, ScikitLearn
using ScikitLearn.CrossValidation: cross_val_score
@sk_import svm: SVC

# Read training matrix
xTrain = CSV.File("./xTrain.csv") |> Tables.matrix

# Read test matrix
xTest = CSV.File("./xTest.csv") |> Tables.matrix

# Read yTrain
yTrain = CSV.File("./yTrain.csv") |> Tables.matrix
yTrain = vec(yTrain)

# Read information about test data ( IDs ).
labelsInfoTest = DataFrame(CSV.File("./sampleSubmission.csv"))

# results = cross_val_score(SVC(kernel="rbf"), xTrain, yTrain, cv=5)
# println("rbf: $(results)")
# results = cross_val_score(SVC(kernel="poly"), xTrain, yTrain, cv=5)
# println("poly: $(results)")

model = SVC(kernel="poly")
model.fit(xTrain, yTrain)

pred = model.predict(xTest)

#Convert integer predictions to character
labelsInfoTest[:,"Class"] = string.(Char.(pred))

#Save predictions
CSV.write("./juliaSVMSubmission.csv", labelsInfoTest)