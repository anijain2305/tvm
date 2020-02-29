scale = 0.9


pdf("money.pdf", height=4.0*scale, width=10*scale)
par(family='Times')
par(oma=c(0,2,0,0))
par(mai=c(0.75,0.60,0.10,0.07))  ##BLTR


sequence = c("resnet18_v1-symbol",
             "resnet50_v1-symbol",
             "resnet50_v1b-symbol",
             "resnet101_v1-symbol",
             "imagenet1k-resnet-152-symbol",
             "inceptionv3-symbol",
             "imagenet1k-inception-bn-symbol",
             "mobilenet1.0-symbol",
             "mobilenetv2_1.0-symbol")

mycols = c()
for (s in sequence) {
    mycols = c(mycols, "blue")
}
mycols = c(mycols, "darkblue")
mycols = c(mycols, "darkblue")

better_names = vector(mode="list", length=length(sequence))
names(better_names) = sequence
better_names[[1]] = "resnet-18"
better_names[[2]] = "resnet-50"
better_names[[3]] = "resnet-50_v1b"
better_names[[4]] = "resnet-101"
better_names[[5]] = "resnet-152"
better_names[[6]] = "inception-v3"
better_names[[7]] = "inception-bn"
better_names[[8]] = "mobilenet-v1"
better_names[[9]] = "mobilenet-v2"

speedup = c()
models = c()
geometric.mean <- function(x){ return(prod(x)^(1/length(x)));}

platforms = c("EC2 C5.12x Intel Cascadelake", "EC2 G4.1x Nvidia GPU")

collect = function(data) {
    fp32_data = data$FP32
    int8_data = data$Int8
    for (s in sequence) {
        row = data[is.element(data$Model, s), ]
        speedup <<- c(speedup, row$Speedup)
        models <<- c(models, better_names[s])
    }
    gmean = geometric.mean(data$Speedup)
    speedup <<- c(speedup, gmean)
    models <<- c(models, "Geomean")
}

collect(read.csv(file = 'c5_fp32_vs_int8.csv'))
speedup <<- c(speedup, 0)
models <<- c (models, "")
collect(read.csv(file = 'g4_fp32_vs_int8.csv'))



ymin = 0.5
ymax = 4.0

cr = speedup
networks = models

barplot(cr, beside=FALSE, ylim=c(ymin,ymax), yaxs='i', yaxt='n', xaxt='n', xpd=FALSE)


yy = seq(ymin, ymax, 0.5)
abline(h=yy, col='black', lty=2)
abline(h=1, col='black')
abline(v=1.2*length(better_names) + 1.2 + 0.7, col='black')
abline(v=1.2*length(better_names) + 1.2 + 0.7, col='black')
abline(v=1.2*length(better_names) + 1.2 + 0.7, col='black')
xx = seq(0, length(networks))*1.2 + 0.5
axis(1, at=xx-0.4, labels=NA)
text(x=xx + 0.5, ymin - 0.12, labels=c(networks, ""), cex=1.00, xpd=NA, srt=40, adj=1)

axis(2, at=yy, label=paste(yy,"x"), las=2, cex.axis=1.0)
mtext("Speedup against TVM FP32", side=2, line=3.2, cex=1.2)
par(new=TRUE)

barplot(cr, beside=FALSE, ylim=c(ymin,ymax), yaxs='i', yaxt='n', xaxt='n', col=mycols, xpd=FALSE)

xx = seq((length(better_names) + 4)/2, length(networks), length(better_names)+ 2.5)
text(x = xx, 3.7, labels=platforms)
print(xx)
box()

dev.off()
