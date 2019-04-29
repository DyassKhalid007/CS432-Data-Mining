#code to generate the correlation matrix
heart = read.csv("heart.csv")
mydata.cor = cor(heart)

#Now plotting barplot of target
counts <- table(heart$target)
barplot(counts, main="Target Barplot", 
        xlab="Number of targets")

#now plotting histogram of sex wrt to target
barplot(table(heart$sex),
        main = "Histogram of sex with respect to target",
        xlab = "Sex",
        ylab = "Target",
        table(heart$target))

