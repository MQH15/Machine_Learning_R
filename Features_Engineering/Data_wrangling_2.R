##########################################################################
# Master Data Science: Data Wrangling 2
##########################################################################

library(janitor)

test_df <- as.data.frame(matrix(ncol = 6))
names(test_df) <- c("firstName", "ábc@!*", "% successful (2009)",
                    "REPEAT VALUE", "REPEAT VALUE", "")

make.names(names(test_df))

test_df %>%
  clean_names()

############### tabyl better than table #################################################

#tabyl() is a tidyverse-oriented replacement for table(). 
#It counts combinations of one, two, or three variables, 
#and then can be formatted with a suite of adorn_* functions to look just how you want. 

# MOtivation:
#table(), leaves much to be desired:

## It doesn't accept data.frame inputs (and thus doesn't play nicely with the %>% pipe)
## It doesn't output data.frames
## Its results are hard to format. Compare the look and formatting choices of an R table 
## to a Microsoft Excel PivotTable or even the table formatting provided by SPSS.

#tabyl() is an approach to tabulating variables that addresses these shortcomings. 
#It's part of the janitor package because counting is such a fundamental part of data 
#cleaning and exploration.

### example 1:

mtcars %>%
  tabyl(gear, cyl) %>%
  adorn_totals("col") %>%
  adorn_percentages("row") %>%
  adorn_pct_formatting(digits = 2) %>%
  adorn_ns() %>%
  adorn_title()


#### example 2: starwars

library(dplyr)
humans <- starwars %>%
  filter(species == "Human")

t1 <- humans %>%
  tabyl(eye_color)

# it also works on vectors and does not count NAs
x <- c("big", "big", "small", "small", "small", NA)
tabyl(x)

t1 %>%
  adorn_totals("row") %>%
  adorn_pct_formatting()

# contingency table or crosstab 

t2 <- humans %>%
  tabyl(gender, eye_color)

t2 %>%
  adorn_percentages("row") %>%
  adorn_pct_formatting(digits = 2) %>%
  adorn_ns()

# 3 variables tabyl
t3 <- humans %>%
  tabyl(eye_color, skin_color, gender)

#If the adorn_ helper functions are called on a list of data.frames 
#- like the output of a three-way tabyl call - they will call purrr::map() 
# to apply themselves to each data.frame in the list:

library(purrr)
humans %>%
  tabyl(eye_color, skin_color, gender, show_missing_levels = FALSE) %>%
  adorn_totals("row") %>%
  adorn_percentages("all") %>%
  adorn_pct_formatting(digits = 1) %>%
  adorn_ns %>%
  adorn_title

#inciso: libreria purrr: map-> reduce

mtcars %>%
  split(.$cyl) %>% # from base R
  map(~ lm(mpg ~ wt, data = .)) %>%
  map(summary) %>%
  map_dbl("r.squared")

# more elegant tables printed using kable from knitr package

humans %>%
  tabyl(gender, eye_color) %>%
  adorn_totals(c("row", "col")) %>%
  adorn_percentages("row") %>% 
  adorn_pct_formatting(rounding = "half up", digits = 0) %>%
  adorn_ns() %>%
  adorn_title("combined") %>%
  knitr::kable()

#You can also call adorn_ functions on other data.frames, 
#not only the results of calls to tabyl(). E.g., 

mtcars %>% adorn_totals("col") %>% adorn_percentages("col") %>% head()

#performs as expected, despite mtcars not being a tabyl.

# example 3:
percent_above_165_cm <- humans %>%
  group_by(gender) %>%
  summarise(pct_above_165_cm = mean(height > 165, na.rm = TRUE))

percent_above_165_cm %>%
  adorn_pct_formatting()

mpg_by_cyl_and_am <- mtcars %>%
  group_by(cyl, am) %>%
  summarise(mpg = mean(mpg)) %>%
  spread(am, mpg)

mpg_by_cyl_and_am


mpg_by_cyl_and_am %>%
  adorn_rounding() %>%
  adorn_ns(
    ns = mtcars %>% # calculate the Ns on the fly by calling tabyl on the original data
      tabyl(cyl, am)
  ) %>%
  adorn_title("combined", row_name = "Cylinders", col_name = "Is Automatic")


###################### other functions

#Explore records with duplicated values for specific combinations of variables with get_dupes()

get_dupes(mtcars, wt, cyl)

# remove_empty() rows and columns

q <- data.frame(v1 = c(1, NA, 3),
                v2 = c(NA, NA, NA),
                v3 = c("a", NA, "b"))
q %>%
  remove_empty(c("rows", "cols"))

#round_half_up()
nums <- c(2.5, 3.5)
round(nums)
round_half_up(nums)

# Count factor levels in groups of high, medium, and low with top_levels()

f <- factor(c("strongly agree", "agree", "neutral", "neutral", "disagree", "strongly agree"),
            levels = c("strongly agree", "agree", "neutral", "disagree", "strongly disagree"))
top_levels(f)

top_levels(f, n = 1)

