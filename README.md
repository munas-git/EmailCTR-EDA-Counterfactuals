# Email CTR Prediction – End‑to‑End ML Pipeline, Analysis Question Answering & Recommendation

## Project Overview

Predicting whether a user will click an email advertisement and uncovering **why** they do so.  The analysis walks through data cleaning, extensive feature engineering, rigorous model comparison and tuning, interpretable ML techniques, and counterfactual what‑if simulations, all in a single, reproducible notebook.

**NOTE**: This entire analysis is part of my weekly series in efforts to **demystify applied statistical techniques through real-world, project-driven examples**, making concepts like propensity modelling, causal inference, and evaluation metrics more accessible to practitioners of all backgrounds.

### Author

**Einstein Ebereonwu** • [GitHub](https://github.com/munas-git) • [LinkedIn](https://www.linkedin.com/in/einstein-ebereonwu/)   
*Dataset: [Kaggle – Click‑Through‑Rate Prediction](https://www.kaggle.com/datasets/swekerr/click-through-rate-prediction)*

---

## Pipeline Highlights

| Stage                            | Key activities                                                                                                                                                                                           |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Data Quality**                 | • Flagged Area Income outliers via box‑plots; retained unless they hurt generalisation.                                                                                                                  |
| **Baseline Feature Engineering** | • Timestamp decomposition → `hour`, `day`, `weekday`, `month`  <br>• Categorical encoding with `LabelEncoder`.                                                                                           |
| **Exploratory Data Analysis**    | • Univariate & bivariate visualisations.<br>• Multicollinearity heat‑map (Age ≈ strongest driver).                                                                                                       |
| **Model Benchmarking**           | • 80/20 stratified split.<br>• **12 classifiers** benchmarked (XGBoost, LightGBM, CatBoost, HistGB, RandomForest …).<br>• Baseline top F1 ≈ **0.875** (XGBoost).                                         |
| **Hyper‑parameter Tuning**       | • `RandomizedSearchCV` (70 iterations).<br>• Best XGBoost: `n_estimators = 600`, `max_depth = 7`, `learning_rate = 0.06`, `subsample = 0.94`, `colsample_bytree = 0.82`.                                 |
| **Model Explainability**         | • **Permutation Importance** + **SHAP TreeExplainer** + partial dependence.<br>• Revealed heavy reliance on `Age`.                                                                                       |
| **Advanced Feature Engineering** | • Mitigated age dominance by introducing:<br>  ◦ `age_group` (bins)<br>  ◦ `part_of_day` (Morning/…/Night)<br>  ◦ `month_part` (Beginning/Middle/End)<br>• Dropped raw `Age`, `hour`.                    |
| **Retraining**                   | • Same tuned XGBoost on engineered set.<br>• F1 ≈ **0.874** (test) with **balanced feature importances**.                                                                                                |
| **Error & Segment Analysis**     | • `qcut`‑based binning on Income, Usage, Time‑on‑Site.<br>• Surfaced under‑performing **topics** (e.g., *Configurable 24/7 Hub*), **cities** (e.g., *Alexandrafort*), **countries** (e.g., *Greenland*). |
| **Counterfactual Analysis**      | • Examined *true negatives* (n = 889).<br>• 461 users could convert with minimal changes: **age‑group reassignment & ad‑topic swap** were most impactful.                                                 |

---

### Tech Stack

Python | pandas | scikit‑learn | XGBoost | LightGBM | CatBoost | SHAP | Matplotlib/Seaborn

---

# Key Findings

### **(EDA-Based)**

1. #### **Q**: ***How does user demographic information (age, income) relate to CTR?***   
  **A**: User demographic features have varying impact on CTR with the following specifications

  - **Age**: CTR increases steadily with age. From as low as ~0.2 from users' in their 20s, rising further between 30s and 40s, and peaking around 0.9 for users in their 50s - 60s.
  - **Income** CTR shows a non-linear relationship with area income, peaking among middle-income areas around $40K, and then gradually declining beyond that point.


2. #### **Q**: ***Do time-based behavior (e.g., daily time spent on site, internet usage, timestamp patterns) influence likelihood of clicking on ads?***  
  **A**: `Yes`, ad-clicks vary by several time-based behavioral factors:  
  - **Daily Time Spent on Site**: Higher CTRs are observed with those who spend longer time on the site, peaking around 70 minutes, with a dip around the 55-minute mark, and after its peak.
  - **Daily Internet Usage**: CTR is highest at moderate usage levels around 140–150, and declines with higher usage.  
  - **Hourly Performance**: CTR steadily increases through the morning and peaks twice `around 10AM, and 12PM`. There's another rise after 8PM.  
  - **Daily Patterns**: CTR is `highly volatile across days`. `Strong performance` is observed at the `start of the month` and again around the `23rd–26th`.   
  - **Weekday Performance**: `Wednesday` stands out with the `highest CTR`, possibly reflecting mid-week browsing intent. `Friday` has the `lowest CTR`, indicating users might be less engaged heading into the weekend.
  - **Monthly Trends**: CTR peaks in April, coinciding with the highest ad volume, then it drops `significantly` in June and July...

3. #### **Q**: ***Are there geographic differences in ad engagement (by city or country)?***   
  **A**: **`Yes`**. Cities such as `New Travistown` and `Westshire`, along with countries like `Tonga` and `Netherlands`, account for the highest CTRs. In contrast, cities like `Gracitown` and `Kingshire`, and countries such as `Singapore` and `Cameroon`, show the lowest recorded CTRs. 

4. #### **Q**: ***Do certain types of 'Ad Topic Line' perform better in terms of CTR?***     
  **A**: **`Yes`**. Ad Topic lines such as `Cloned Object-Oriented Benchmark`, `Innovative Interactive Portal`, etc., are amongst the top performers, while `Front-Line Dynamic Model` and `Inverse Local Hub` are associated with the lowest clicks.

5. #### **Q**: ***What is the relationship between internet use and time on site, and how do those correlate with CTR?***     
  **A**: Even though they individually have their impact, the interaction between internet usage and time spent on site doesn't show a strong pattern of correlation with ad click behavior.

### **(Post Model-Training)**

6. #### **Q**: ***Which user attributes (age, income, location...) are most influential in driving ad clicks, according to the predictive model?***   

  **A**: Based on final model trained, the most important features are:
  
- **Age Group**: The strongest driver of ad clicks. Older users are significantly more likely to click compared to younger ones. The influence of age is both large and consistent.
- **Location (Country & City)**: Where a user lives plays a major role. Some countries and cities show notably higher or lower engagement rates. These effects can vary sharply between regions.  
- **Ad Header (Ad Topic Line)**: The content or theme of the ad is a top factor in whether users click. Some topics resonate much more strongly with certain audiences.
- **Income Level (Area Income)**: Users from higher-income areas show a moderate increase in click likelihood. While not the top driver, income consistently influences ad engagement.
- **Internet Usage Behavior**: Daily internet habits matter. Users with moderate usage are more likely to engage than those with very low or very high usage.
- **Time on Site**: People who spend more time on the site are generally more likely to click, though this effect varies.
- **Day of the Week**: The specific day matters more than whether it’s a weekday or weekend. Some days consistently perform better than others.

7. #### **Q**: ***Which feature groups are most associated with unpredictable or inaccurate model predictions?***

**A**: The model demonstrates lower predictive accuracy across certain `topics`, `cities`, and `countries`, which may indicate inconsistent user behavior or limited training data coverage.

- **Topics with Most Unreliable Predictions**: Assimilated Multi-State Paradigm, Business-Focused Value-Added Definition, Configurable 24/7 Hub, Configurable Dynamic Secured Line, Customer-Focused Fault-Tolerant Implementation 

- **Cities with Poor Predictive Performance**: Alexandrafort, Bakerhaven, Crawfordfurt, East Ericport, Evansville

- **Countries with Weak Model Accuracy**: Greenland, Italy, South Georgia, Thailand, Turks and Caicos Islands


8. #### **Q**: **Which types of users are most responsive to changes in key factors?**

**A**: The counterfactual analysis highlights that certain user segments exhibit higher responsiveness to changes in key targeting factors, offering opportunities to refine audience segmentation and campaign focus:

- **Age Groups 0–45 Show Notable Responsiveness**:  
  Users in the `0–25`, `26–35`, and `36–45` brackets demonstrate improved predicted engagement when hypothetically associated with characteristics of older, more responsive age groups (such as `46–55`). This indicates that these younger users have latent responsiveness that can potentially be activated through targeted messaging or expanded delivery.

- **Users React Strongly to Specific Ad Topics**:  
  Exposure to certain high-performing ad topics, especially themes like **"Visionary Mission-Critical Application"** is associated with increased engagement, particularly among users initially exposed to less effective topics (e.g., `"Sharable Reciprocal Project"`, `"Managed Client-Server Access"`). This suggests that tailoring content toward impactful themes can enhance user responsiveness.

- **Timing Factors Show Subtle Influence**:  
  Some users’ predicted engagement varies with the timing of ad delivery (day of week, month), though this effect is less pronounced compared to age and topic factors.

- **Gender Has Minimal Impact on Responsiveness**:  
  Changes in gender targeting show little effect on engagement, indicating gender may not be a critical dimension for refining audience segments in this context.


9. #### **Q**: **What changes in timing or targeting can boost engagement among the least responsive users?**

**A**: Counterfactual analysis identified `461 out of 889` users for whom targeted adjustments could increase predicted click probability to `>= 0.5`. These findings suggest practical steps to boost engagement among less responsive segments:

- **Retarget Toward More Responsive Age Groups**: Many counterfactuals indicate that users in younger age brackets, such as `Age 0–25, 26–35, and 36–45`, would have been more likely to convert if they belonged to the `Age 46–55` group. While age itself cannot be changed, this suggests that `delivering the same ads to older age groups could substantially improve conversions`. For instance, 133 cases involved shifting from 26–35 to 46–55, and 85 from 0–25 to 46–55, highlighting the strong engagement of the 46–55 demographic. There are also smaller gains when targeting 36–45 instead of younger groups.

- **Refine Ad Messaging Toward High-Performing Themes**: Several underperforming ad topics, such as `"Sharable Reciprocal Project"` or `"Managed Client-Server Access"`, saw improved predicted engagement when changed to `"Visionary Mission-Critical Application"`. This theme appears to resonate broadly with disengaged users, with notable counts (12 and 8 instances respectively) supporting this shift.

- **Adjust Ad Delivery Timing (Month and Weekday)**: Although changes here affected fewer cases, shifting ad delivery from months like `January to July`, or `July to February`, showed small but measurable improvements. Similarly, adjusting delivery days from less engaging weekdays like `Friday or Tuesday` to more engaging days such as `Monday or Wednesday` was associated with better predicted outcomes. While these adjustments are less impactful overall, they could still provide incremental gains if incorporated strategically.

- **Minimal Impact from Gender-Based Targeting**: Only one case suggested benefit from switching gender targeting (Male to Female), indicating that gender may not be a primary lever for improving engagement in this dataset.


10. #### **Q**: **What hidden trends or unexpected feature combinations did the model uncover that could inform more effective marketing strategies moving forward?**

**A**: The model revealed several nuanced and non-obvious interactions between features that can significantly influence ad engagement, highlighting opportunities for more sophisticated and personalised marketing approaches:

- **Geography Modulates Age Effects**:  
  The impact of age on ad clicks varies substantially by city. For instance, younger users in some cities exhibit engagement patterns similar to older users elsewhere. This suggests that **local cultural, economic, or social factors shape how different age groups respond**, implying that a one-size-fits-all age targeting strategy may miss key regional subtleties.

- **Ad Topic Lines Show Age-Specific Appeal**:  
  Certain ad themes resonate differently across age groups, an ad topic that performs well for middle-aged users may underperform among younger or older audiences. This calls for `dynamic content personalisation` where ad creatives are tailored not just broadly by age but fine-tuned to age-topic synergies.

- **Internet Usage Intensity Interacts with Age**:  
  Time spent online influences click likelihood differently depending on a user’s age. For example, moderate internet usage might predict high engagement for one age cohort but lower engagement for another. This finding enables `targeting users based on a combined profile of age and digital behavior`, optimising ad delivery to the most receptive segments.

- **Temporal Patterns Vary by Age Group**:  
  The day of the week impacts responsiveness unevenly across age groups, with some groups showing heightened engagement on specific days. Recognising these `age-specific temporal trends` allows marketers to schedule campaigns with greater precision, maximising impact by aligning ad delivery with when target users are most receptive.

---

### Recommendations for Optimised Targeting and Engagement

- **Prioritise Age-Based Segmentation:**  
  Focus campaigns on age groups showing strong responsiveness, especially the 46–55 demographic, while tailoring messaging for younger cohorts to enhance engagement.

- **Leverage High-Impact Ad Topics:**  
  Concentrate creative efforts on proven, resonant themes (e.g., `Visionary Mission-Critical Application1) to maximise user interest and conversion.

- **Adopt Multi-Dimensional Segmentation:**  
  Integrate factors such as geography, age, and digital behavior for more precise audience targeting rather than isolated criteria.

- **Personalise Messaging and Creative Testing:**  
  Develop and test tailored content for specific age-topic combinations to boost relevance and engagement rates.

- **Refine Campaign Scheduling:**  
  Use insights on age-related temporal patterns to optimise ad delivery timing, ensuring ads reach users when they are most receptive.
