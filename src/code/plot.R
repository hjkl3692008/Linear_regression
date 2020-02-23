library(readr)

# read data
data <- read_csv("C:/Users/79109/gm/workplace/Linear_regression/src/data/kc_house_data.csv")

# plot sqft_living
ggplot(data,aes(x=sqft_living))+
  geom_histogram(col = 'black')+
  coord_cartesian(ylim = c(0,10))+
  scale_x_continuous(breaks = seq(0,15000,4000))
 
 
# plot scatter points x=sqft_living,y=price
options(repr.plot.width=6, repr.plot.height=3)
ggplot(data,aes(x=sqft_living,y=price,col=price))+
  geom_point(position = "jitter")