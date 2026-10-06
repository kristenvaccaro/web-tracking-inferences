import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def generateData():
    # Set random seed for reproducibility
    np.random.seed(42)
    n_samples = 10000
    
    # -----------------------------------------------------------------------------
    # 1. Base Feature Generation
    # -----------------------------------------------------------------------------
    
    # Demographics
    age = np.random.randint(18, 72, size=n_samples)
    
    edu_levels = ['High School', 'Associate', 'Bachelor', 'Master', 'PhD']
    edu_weights = [0.25, 0.20, 0.35, 0.15, 0.05]
    education = np.random.choice(edu_levels, size=n_samples, p=edu_weights)
    
    zip_codes = ['90210', '10001', '60611', '98101', '75001', '30301', '48201', '02113']
    zip_wealth_map = {
        '90210': 1.6, '10001': 1.4, '60611': 1.3, '98101': 1.2,
        '02113': 1.1, '75001': 1.0, '30301': 0.9, '48201': 0.7
    }
    zip_code = np.random.choice(zip_codes, size=n_samples)
    zip_wealth = np.array([zip_wealth_map[z] for z in zip_code])
    
    # Device & Technical Data
    device_models = ['Budget Android', 'Mid-Tier Android', 'Flagship Android', 'Older iPhone', 'Flagship iPhone']
    device_weights = [0.20, 0.30, 0.15, 0.15, 0.20]
    device_model = np.random.choice(device_models, size=n_samples, p=device_weights)
    
    device_tier_map = {
        'Budget Android': 0.7, 'Mid-Tier Android': 0.9,
        'Older iPhone': 1.0, 'Flagship Android': 1.3, 'Flagship iPhone': 1.5
    }
    device_tier = np.array([device_tier_map[d] for d in device_model])
    
    resolution_map = {
        'Budget Android': '720x1600',
        'Mid-Tier Android': '1080x2400',
        'Flagship Android': '1440x3200',
        'Older iPhone': '1170x2532',
        'Flagship iPhone': '1290x2796'
    }
    screen_resolution = [resolution_map[d] for d in device_model]
    
    connection_qualities = ['3G', '4G/LTE', '5G', 'Wi-Fi Broadband']
    conn_weights = [0.08, 0.42, 0.30, 0.20]
    connection_quality = np.random.choice(connection_qualities, size=n_samples, p=conn_weights)
    
    conn_tier_map = {'3G': 0.8, '4G/LTE': 1.0, '5G': 1.2, 'Wi-Fi Broadband': 1.3}
    conn_tier = np.array([conn_tier_map[c] for c in connection_quality])
    
    # Behavioral Data
    categories = ['Electronics', 'Fashion', 'Home & Living', 'Beauty & Personal Care', 'Sports & Outdoors', 'Books & Media']
    product_categories_viewed = np.random.choice(categories, size=n_samples)
    categories_prior_purchases = np.random.choice(categories, size=n_samples)
    
    num_prior_orders = np.random.negative_binomial(5, 0.3, size=n_samples) + 1
    monetary_value_prior_orders = num_prior_orders * np.random.uniform(25, 140, size=n_samples)
    
    # Social Media & App Data
    app_categories = ['Social Media', 'E-Commerce', 'Gaming', 'Finance & Investing', 'Productivity', 'Health & Fitness']
    types_apps_downloaded = np.random.choice(app_categories, size=n_samples)
    social_likes_count = np.random.poisson(lam=130, size=n_samples)
    
    # -----------------------------------------------------------------------------
    # 2. Correlated Target Feature Engineering
    # -----------------------------------------------------------------------------
    
    # Household Income: Dependent on age, education, zip code, monetary value, device model, and connection quality
    edu_tier_map = {'High School': 1.0, 'Associate': 1.15, 'Bachelor': 1.45, 'Master': 1.75, 'PhD': 2.1}
    edu_tier = np.array([edu_tier_map[e] for e in education])
    
    # Peak income around age 48-52
    age_factor = 1.0 - 0.0007 * (age - 50)**2
    
    base_income = (
        22000
        + 32000 * edu_tier * zip_wealth * age_factor
        * (0.6 + 0.4 * device_tier)
        * (0.85 + 0.15 * conn_tier)
    )
    monetary_boost = monetary_value_prior_orders * 4.5
    income_noise = np.random.normal(0, 10000, size=n_samples)
    
    household_income = base_income + monetary_boost + income_noise
    household_income = np.clip(household_income, 0, 250000)
    
    # Average Basket Size: Dependent on household income and order history
    avg_basket_size = 1 + np.round(
        (household_income / 32000) + (num_prior_orders * 0.12) + np.random.normal(0, 0.7, size=n_samples)
    )
    avg_basket_size = np.clip(avg_basket_size, 1, 15).astype(int)
    
    # Average Basket Value: Monetary value divided by prior order count
    avg_basket_value = (monetary_value_prior_orders / num_prior_orders) * (0.85 + 0.3 * (household_income / 100000))
    
    # Customer Lifetime Value (CLV): Derived from spend, frequency, and projected income tier
    customer_lifetime_value = (
        monetary_value_prior_orders * 1.6 
        + (household_income * 0.04) 
        + np.random.normal(0, 150, size=n_samples)
    )
    customer_lifetime_value = np.clip(customer_lifetime_value, 50, None)
    
    # Ad Conversion Label (Target for Classification models)
    conversion_logits = (
        -2.0
        + (household_income / 100000) * 0.8
        + (social_likes_count / 200) * 0.5
        + (device_tier - 1.0) * 0.6
    )
    conversion_prob = 1 / (1 + np.exp(-conversion_logits))
    ad_clicked = np.random.binomial(1, conversion_prob)
    
    # -----------------------------------------------------------------------------
    # 3. Assemble Master DataFrame
    # -----------------------------------------------------------------------------
    
    df = pd.DataFrame({
        'user_id': [f"USR_{10000 + i}" for i in range(n_samples)],
        'age': age,
        'education': education,
        'zip_code': zip_code,
        'household_income': np.round(household_income, 2),
        'device_model': device_model,
        'screen_resolution': screen_resolution,
        'connection_quality': connection_quality,
        'product_categories_viewed': product_categories_viewed,
        'categories_prior_purchases': categories_prior_purchases,
        'num_prior_orders': num_prior_orders,
        'monetary_value_prior_orders': np.round(monetary_value_prior_orders, 2),
        'avg_basket_size': avg_basket_size,
        'avg_basket_value': np.round(avg_basket_value, 2),
        'customer_lifetime_value': np.round(customer_lifetime_value, 2),
        'types_apps_downloaded': types_apps_downloaded,
        'social_likes_count': social_likes_count,
        'ad_clicked': ad_clicked
    })
    
    # -----------------------------------------------------------------------------
    # 4. Train / Test Split (70% Train, 30% Test)
    # -----------------------------------------------------------------------------
    
    X = df.drop(columns=['ad_clicked'])
    y = df['ad_clicked']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    # print(f"Full Dataset Shape: {df.shape}")
    # print(f"Training Set Shape: {X_train.shape}")
    # print(f"Testing Set Shape:  {X_test.shape}\n")
    
    # # Sample records
    # print("--- Sample Records ---")
    # print(df.head(3).T)
    
    # # Verification of correlations
    # print("\n--- Correlation Matrix of Derived Features ---")
    # print(df[['household_income', 'age', 'monetary_value_prior_orders', 'avg_basket_size', 'avg_basket_value', 'customer_lifetime_value']].corr())
    
    return X_train
