# Before we begin:

### When Do We Split Data Into Train and Test Set?

In general, the split between train and test sets should occur **before** any analysis that could influence the model's performance. I split the data after EDA. Here are guidelines on when to split:

#### 1. EDA (Exploratory Data Analysis)
- **Before split:** Okay, if distributions are assumed to be similar. Usually, we assume the distribution of test and train set are similar. If the distributions differ substantially, the model may suffer from data drift or poor generalization.
- **After split:** Preferred, to avoid bias.

#### 2. Imputation and Missing Value Handling
- **Before split:** Constant values like `0`. Using mean, median of whole data set might cause data leakage.
- **After split:** Imputation using training data avoids leakage.

#### 3. Outlier Handling
- **Before split:** If expected in real-world input.
- **After split:** Ideal to avoid bias, ensures harder-to-predict data remains for testing.

#### 4. Feature Encoding
- **Before split:** Safe for One-Hot Encoding (OHE); avoid target encoding (causes leakage), frequency encoding etc.
- **After split:** Avoid leaks by using only training data.

#### 5. Feature Engineering and Scaling
- **Before split:** Safe if performed within a single observation (e.g., adding or subtracting features).
- **After split:** FAlways fit scalers - StandardScaler(), MinMaxScaler() - to the training data, then apply to both train (fit_transform) and test sets (transform).

#### 6. Feature Selection
- **Before split:** Can cause leakage.
- **After split:** Use training data to avoid bias.


In short, it is important to **avoid insights from the test data** affecting our decisions, ensuring a fair and unbiased model evaluation.