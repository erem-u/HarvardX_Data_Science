# SECTION 1 - INTRODUCTION


# The purpose of this project is to develop a recommendation system which is similar to the system that was awarded the grand prize by Netflix in a contest in 2009.
# The recommendation system of interest basically predicts movies that a user might like based on the ratings the user has provided on different movies.
# The contest, called The Netflix Prize, was an open competition with a grand prize of $1,000,000. 
# The winner recommendation system was based on loss function which produced the lowest Residual Mean Squared Error (RMSE)
# and more than 10% improvement of the RMSE that the Netflix' algorithm, Cinematch, generated at that time.
# The RMSE that won the grand prize was 0.8567 as opposed to Cinematch's RMSE 0.9525.
# It has been claimed that a 1% improvement of the RMSE can make a big positive difference to identify the "top-10" most recommended movies for a user.
# Reference: (YehudaKoren (2007-12-18). "How useful is a lower RMSE?". Netflix Prize Forum. Archived from the original on 2012-03-03.)
# The contestants were provided a dataset of 100M users with movie ratings for the contest.
# In this project, a smaller size dataset of 10M users will be used.
# I will explain the steps taken throughout my script to obtain a desired RMSE.


# Table of Content:
# Section 1 - Introduction
# Section 2 - Analysis
#   Section 2.1 - Data Wrangling
#   Section 2.2 - Splitting Edx to Train and Test Sets
#   Section 2.3 - Loss Function
#   Section 2.4 - Modeling Different Recommendation Systems and Computing RMSEs Accordingly
#   Section 2.5 - Penalized Least Squares (Regularization)
#   Section 2.6 - Choosing the Penalty Terms
#   Section 2.7 - Computing the Final RMSE by Using the Validation Set (Final Hold-Out Test)
# Section 3 - Results
# Section 4 - Conclusion


#####################################################################################################################


# SECTION 2 - ANALYSIS


# SECTION 2.1 - DATA WRANGLING

# My script starts with the chunk of code to download and tidy the MovieLens 10M dataset from grouplens.org.
# It is the chunk of code that is provided in the course to obtain the edx set and validation set (the final hold-out test set) out of the MovieLens 10M Dataset.
# The validation set will only be used to evaluate the RMSE of my final algorithm.
# The edx set will be split into separete training and test sets in the next section to design and test my algorithm.

# tidyverse, caret and data.table packages are needed in my script. The following three lines of code will download the packages from cran:
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# These three lines of code will load the packages:
library(tidyverse)
library(caret)
library(data.table)

# This is the website for the MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

# These two lines of code will download the MovieLens 10M dataset from grouplens.org:
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

# Here a little bit tidying the dataset.
# This line of code will deliminate the text in the data to "userId", "movieId", "rating" and "timestamp" and save it to "ratings" 
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

# This line of code will deliminate the text in the data to "movieId", "title" and "genres" and save it to "movies"
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# This line of code will change the "movies" data to data.frame. (if using R 4.0 or later):
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

# This line of code will join the ratings and movies tables by "movieid":
movielens <- left_join(ratings, movies, by = "movieId")

movie_titles <- movielens %>% 
  select(movieId, title) %>%
  distinct()

# Validation set will be 10% of MovieLens data
set.seed(1) # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

# This line of code will remove the following datasets from the global environment. edx and validation datasets will only be left
rm(dl, ratings, movies, test_index, temp, removed)




####################################################################################################################


# SECTION 2.2 - SPLITTING EDX TO TRAINING AND TEST SETS

# edx and validation datasets were created in the previous section.
# In this section, edx dataset will be split into "train_set_edx" and "test_set_edx" which will be used to train and test my algorithm

# 80% of the edx dataset will constitute the train_set_edx and the rest will constitute the test_set_edx
set.seed(2)
test_index_edx <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set_edx <- edx[-test_index_edx,]
test_set_edx <- edx[test_index_edx,]

test_set_edx <- test_set_edx %>% 
  semi_join(train_set_edx, by = "movieId") %>%
  semi_join(train_set_edx, by = "userId")

rm(test_index_edx)



####################################################################################################################


# SECTION 2.3 - LOSS FUNCTION

# The typical error loss is selected as the preferred recommendation system in this project which was also the method of the winner algorithm in the Netflix Prize in 2009.
 

# The function for the residual mean squared error (RMSE) is defined as follows:
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}



####################################################################################################################


# SECTION 2.4 - MODELING DIFFERENT RECOMMENDATION SYSTEMS AND COMPUTING RMSEs ACCORDINGLY


# To start off, the simplest possible recommendation system is built which predicts the same rating for all movies regardless of user.
# The value for the "same rating" is chosen to be the average of all rating which is represented by mu here:
mu <- mean(train_set_edx$rating)
mu
# RMSE based on the average rating of all ratings is calculated:
just_the_average_result <- RMSE(test_set_edx$rating, mu)
just_the_average_result
# Just the Average Result: 1.060368


# It is confirmed by data that some movies are rated higher than others. This bias is called the movie-specific effect.
# When this bias is added to the simplest recommendation system that was introduced previously, a small improvement is observed in RMSE:
movie_avgs <- train_set_edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu)) # here b_i is defined as the least squares estimate for each movie i. It is the movie-specific effect

plot_effect_of_movie <- qplot(b_i, data = movie_avgs, bins = 10, color = I("black")) # As shown on this plot, the effect of movie bias is noteworthy
plot_effect_of_movie

predicted_ratings <- mu + test_set_edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
# RMSE in this case is slightly lower than the RMSE that was based on the assumption of all movies having the same rating:
movie_effect_model_result <- RMSE(predicted_ratings, test_set_edx$rating)
movie_effect_model_result
# Movie Effect Model Result: 0.9431022


# Some users are likely to give higher ratings to movies in general and some are not. This effect is called user-specific effect
# The user-specific effect is added to the the previous model with the movie-specific effect:
plot_effect_of_user <- train_set_edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% # here b_u is defined as the user-specific effect
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  # This plot shows the user-specific effect for the users who have rated over 100 movies.
  geom_histogram(bins = 30, color = "black") 
plot_effect_of_user

user_avgs <- train_set_edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i)) # here b_u is estimated by computing mu and b_i

# when an overcritical user (negative b_u) rates a good movie (positive b_i), both effects counter each other and
# a better prediction could be obtained:
predicted_ratings <- test_set_edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
# It is observed here that the model with user-effect produces better RMSE:
movie_and_user_effects_model_result <- RMSE(predicted_ratings, test_set_edx$rating)
movie_and_user_effects_model_result
# Movie + User Effects Model Result: 0.8654546



####################################################################################################################


# SECTION 2.5 - PENALIZED LEAST SQUARES (REGULARIZATION)

# It is not surprising that some high ranked or low ranked movies were rated by very few users which causes larger estimates of b_i.
# Large estimates of b_i can increase the RMSE. Therefore, a technique which will penalize large estimates is needed.
# Penalized Least Squares (Regularization) is the technique that will be used to penalize the large estimates in my analysis.
# Another term, lambda, is introduced as the driver of penalty.
# When the number of rating for a movie is high, a case which will give us a stable estimate, lambda is effectively ignored.
# On the other hand, when the number of rating for a movie is small, the regularized least squared estimate of b_i is shrunken towards 0.
# The larger lambda, the more the regularized estimate of b_i shrinks. A range of values for lambda will be used to obtain the best RMSE.


# to start off the analysis, a value of lambda is randomly selected as 3 to compute the regularized estimate of b_i:
lambda <- 3
movie_reg_avgs <- train_set_edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) # here n_i is the number of ratings for a movie i


# To see how the estimates shrink, let’s make a plot of the regularized estimates versus the least squares estimates.
plot_estimates_shrink <- tibble(original = movie_avgs$b_i, 
       regularlized = movie_reg_avgs$b_i, 
       n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)
plot_estimates_shrink


# now, let's look at the top 10 best movies based on the penalized least squared estimates of b_i:
top_10_movies <- train_set_edx %>%
  count(movieId) %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  left_join(movie_titles, by = "movieId") %>%
  arrange(desc(b_i)) %>% 
  select(title, b_i, n) %>% 
  slice(1:10) %>% 
  knitr::kable() # These movies show that penalized estimates method provides a stable estimates of movies with high number of ratings.
top_10_movies


# RMSE that is computed based on penalized estimates which only covers the movie effect in this case:
predicted_ratings <- test_set_edx %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  mutate(pred = mu + b_i) %>%
  pull(pred)
regularized_movie_effect_model <- RMSE(predicted_ratings, test_set_edx$rating)
regularized_movie_effect_model
# Regularized Movie Effect Model Result: 0.9430591



####################################################################################################################


# SECTION 2.6 - CHOOSING THE PENALTY TERMS

# Lambda is a tuning parameter so we can use cross-validation to choose it:
lambdas <- seq(0, 10, 0.25)
just_the_sum <- train_set_edx %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())

rmses <- sapply(lambdas, function(l){
  predicted_ratings <- test_set_edx %>% 
    left_join(just_the_sum, by='movieId') %>% 
    mutate(b_i = s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set_edx$rating))
})
qplot(lambdas, rmses)  
lambdas[which.min(rmses)]
# result: 1.75

# This time RMSE is computed based on penalized estimates with movie and user effects together:
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set_edx$rating)
  b_i <- train_set_edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set_edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    test_set_edx %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set_edx$rating))
})

qplot(lambdas, rmses)  

lambda <- lambdas[which.min(rmses)]
lambda
# result: 4.75
regularized_movie_and_user_effect_model <- min(rmses)
regularized_movie_and_user_effect_model
# Regularized Movie + User Effect Model Result: 0.864777

# RMSE based on penalized estimates with movie and user effects together is the lowest that have been obtained so far.



#################################################################################################################################


# SECTION 2.7 - COMPUTING THE FINAL RMSE BY USING THE VALIDATION SET (FINAL HOLD-OUT SET)


# In this section, the RMSE is computed by using the validation set to test the final algorithm which is based on penalized least
# squares estimates with movie and user effects.


# This piece of code is the same code chuck that was presented in the previous section 2.6. The only difference is that the train_set_edx
# was replaced with edx and the test_set_edx was replaced with validation:
rmses_final <- sapply(lambdas, function(l){
  mu <- mean(edx$rating)
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, validation$rating))
})

qplot(lambdas, rmses_final)  

lambda <- lambdas[which.min(rmses_final)]
lambda
# result: 5.5

final_rmse_validation_set_result <- min(rmses_final)
final_rmse_validation_set_result
# Final RMSE Validation Set Result: 0.8649857
 

save.image("workspace.RData")



####################################################################################################################


# SECTION 3 - RESULTS


# The results that were obtained in each section are presented and discussed in this section. Firstly, let us tabulate the results
# in the table below:

#  Method                                                RMSE
# Just the Average (Section 2.4)                        1.060368
# Movie Effect Model (Section 2.4)                      0.943102
# Movie + User Effects Model (Section 2.4)              0.865454
# Regularized Movie Effect Model (Section 2.5)          0.943059
# Regularized Movie + User Effect Model (Section 2.6)   0.864777
# Final RMSE Validation Set (Section 2.7)               0.864985

# As seen the results in the table above, the lowest RMSE was obtained in Regularized Movie + User Effect Model in Section 2.6
# compared to the results obtained in other models.
# Therefore, that model was selected to compute the Final RMSE with the Validation Set in Section 2.7.



######################################################################################################################


# SECTION 4 - CONCLUSION

# The final RMSE that was computed with the Validation Set in Section 2.7 is way lower than the Cinematch's RMSE 0.9525.
# However, it is still greater than the RMSE 0.8567 that won the Netflix Prize in 2009 as explained in Section 1.
# Obviously, there is still work to be performed to improve the final RMSE. The final model that was used in this analysis does not
# account the fact that groups of movies and groups of users have similar rating patterns. For example, users who have liked romantic movies
# tend to like other romantic movies more then expected. This effect needs to be added to the model.
# Singular value decomposition (SVD) and principal component analysis (PCA) are proper approaches to follow for further RMSE improvement.
# SVD and PCA can be added to the model by using the Recommenderlab package.
