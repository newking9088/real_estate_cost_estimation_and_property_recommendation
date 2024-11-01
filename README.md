# real_estate_cost_estimation_and_property_recommendation
This project aimed to enhance property recommendations by predicting prices with high accuracy. Utilizing KNN and collaborative filtering algorithms, it achieved a 95% accuracy rate and a 5% margin of error, effectively tailoring property suggestions to customer preferences through the analysis of historical pricing data and user behaviors.

## Quick Link
[Blogpost](https://nycdatascience.com/blog/student-works/end-to-end-machine-learning-pipeline-for-real-estate-valuation-recommendation-engine/?aiEnableCheckShortcode=true)

## Robust Real Estate Valuation Model 🏡🧮

- Developed a highly accurate property valuation model with 95% accuracy, able to estimate prices within a 5% margin of error.
- Leveraged an ensemble of advanced machine learning algorithms including CatBoost, LightGBM, Random Forest, and AdaBoost.
- Achieved excellent performance metrics with R² scores above 0.90 on validation data.


## Automated Machine Learning Pipeline 🤖🔍

- Implemented a streamlined end-to-end ML pipeline for data ingestion, preprocessing, feature engineering, model training, and hyperparameter tuning.
- Ensured data integrity and prevented leakage by carefully sequencing preprocessing steps and storing pipeline parameters.
- Enabled seamless deployment by persisting the preprocessing pipeline and trained models.

## Comprehensive Feature Engineering and Selection 🔧🔍

- Derived new domain-specific features like house age, total square footage, number of bathrooms, and years since last remodel.
- Performed rigorous feature selection using statistical techniques like f_regression and correlation analysis to identify the most important predictors.
- Analyzed the associations between categorical features using Chi-square and Cramer's V to select the most significant ones.


## Intelligent Property Recommendation Engine 🔍🏠

- Implemented a nearest-neighbor based recommendation system to match properties based on user preferences and property characteristics.
- Leveraged the robust data transformation pipeline to create a rich feature space for property matching.
- Provided real-time property recommendations by efficiently computing multidimensional feature similarity.

## Insights into Price Drivers 💰📈

- Analyzed feature importance using the CatBoost model, revealing that 'TotalSqFt' and 'OverallQual' are the most influential factors, accounting for nearly 50% of the price impact.
- Identified other key drivers like 'TotalBaths', 'YrRemodAge', and 'HouseAge' that significantly influence property prices.
