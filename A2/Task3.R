spambase = read.csv("spambase_csv.csv")

#This line of code split the data into group of 4 named as "1","2","3","4"
splitData = split(spambase,c("1","2","3","4"))

#Now constructing the corelation matrix of each attribute
cor1 = cor(splitData[[1]])

cor2 = cor(splitData[[2]])

cor3 = cor(splitData[[3]])

cor4 = cor(splitData[[4]])


 