# Datasets

This folder contains datasets used in the project for recommendation systems and customer behavior analysis.

## 1. Personalized Recommendation Systems Dataset

**Source:** [Download Link](https://www.kaggle.com/datasets/alfarisbachmid/personalized-recommendation-systems-dataset?resource=download)

**Description:**  
A dataset for personalized recommendation system analysis, containing user-item interactions, ratings, platform usage, and regional information.

**Features:**
- `User_ID` — unique identifier for each user
- `Item_ID` — unique identifier for each item
- `Category` — type of item interacted with
- `Rating` — user rating from 1.0 to 5.0
- `Timestamp` — date and time of interaction
- `Price` — item price in USD
- `Platform` — platform or device used for interaction
- `Location` — geographic region of the user

---

## 2. Supermarket Dataset for Predictive Marketing 2023

**Source:** [Download Link](https://www.kaggle.com/datasets/hunter0007/ecommerce-dataset-for-predictive-marketing-2023)

**Description:**  
A large supermarket consumer behavior dataset designed for predictive marketing tasks, including order history, product information, and reorder behavior.

**Features:**
- `order_id` — unique order identifier
- `user_id` — unique user identifier
- `order_number` — number of the order
- `order_dow` — day of the week the order was made
- `order_hour_of_day` — time of the order
- `days_since_prior_order` — time since previous order
- `product_id` — unique product identifier
- `add_to_cart_order` — order in which items were added to cart
- `reordered` — indicates whether the product was reordered
- `department_id` — unique identifier for each department
- `department` — department name
- `product_name` — product name

---

## 3. Shopping Behavior & Preferences Study

**Source:** [Download Link](https://www.kaggle.com/datasets/ranaghulamnabi/shopping-behavior-and-preferences-study)

**Description:**  
A dataset focused on shopping behavior and customer preferences, useful for purchase pattern analysis and customer segmentation.

**Note:**  
This dataset is **not included in the repository files** because it is too large. It is available only through the download link above.

**Features:**
- `Customer ID`
- `Age`
- `Gender`
- `Item Purchased`
- `Category`
- `Purchase Amount (USD)`
- `Location`
- `Size`
- `Color`
- `Season`
- `Review Rating`
- `Subscription Status`
- `Shipping Type`
- `Discount Applied`
- `Promo Code Used`
- `Previous Purchases`
- `Payment Method`
- `Frequency of Purchases`

---

## 4. E-commerce Customer Data For Behavior Analysis

**Source:** [Download Link](https://www.kaggle.com/datasets/shriyashjagtap/e-commerce-customer-for-behavior-analysis)

**Description:**  
A customer behavior analysis dataset containing purchase, return, and churn information for e-commerce customers.

**Features:**
- `Customer ID` — unique identifier for each customer
- `Customer Name` — generated customer name
- `Customer Age` — generated customer age
- `Gender` — generated customer gender
- `Purchase Date` — date of purchase
- `Product Category` — category of purchased product
- `Product Price` — price of the purchased product
- `Quantity` — quantity purchased
- `Total Purchase Amount` — total amount spent in transaction
- `Payment Method` — method of payment
- `Returns` — whether the product was returned
- `Churn` — whether the customer churned

---

## 5. H&M Personalized Fashion Recommendations

**Source:** [Competition Data](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)

**Description:**  
A large-scale fashion recommendation dataset from the H&M Kaggle competition, designed for personalized product recommendation based on historical transactions, customer metadata, and product metadata.

**Note:**  
This dataset is **not included in the repository files** because of its size and competition-based distribution. It should be downloaded manually from Kaggle.

**Main files and features:**
- `articles.csv` — product metadata with fields such as `article_id`, `prod_name`, `product_type_name`, `colour_group_name`, `department_name`, `section_name`, `garment_group_name`, `detail_desc`
- `customers.csv` — customer metadata including `customer_id`, `FN`, `Active`, `club_member_status`, `fashion_news_frequency`, `age`, `postal_code`
- `transactions_train.csv` — historical purchase log with `t_dat`, `customer_id`, `article_id`, `price`, `sales_channel_id`
- `sample_submission.csv` — sample recommendation submission format
- `images/` — product images linked to articles

**Why it is useful:**
- suitable for recommendation systems and cold-start experiments;
- combines transactional, customer, catalog, and image information;
- supports customer behavior analysis, candidate generation, ranking, and hybrid recommendation approaches.

---

## Notes

- All datasets are provided in CSV format.
- The datasets are intended for experiments in recommendation systems, customer behavior analysis, and predictive modeling.
- Some datasets are not stored in the repository due to file size limitations and must be downloaded manually.
