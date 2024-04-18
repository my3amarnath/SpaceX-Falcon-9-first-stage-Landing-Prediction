# SpaceX Falcon 9 first stage Landing Prediction
IBM Capstone Data Science project that predicts the success rate of landing SpaceX's Falcon 9 first stage by training a machine learning model and use public information to predict if SpaceX will reuse the first stage.

SpaceX advertises Falcon 9 rocket launches on its website, with a cost of 62 million dollars; other providers cost upward of 165 million dollars each, much of the savings is because SpaceX can reuse the first stage. Therefore if we can determine if the first stage will land, we can determine the cost of a launch. This information can be used if an alternate company wants to bid against SpaceX for a rocket launch. In this project we will determine what is the likelihood that SpaceX reuse the first stage by training a machine learning model. And determine what factors are used to predict if the first stage will land successfully.

In this project we will determine what is the likelihood that SpaceX reuse the first stage by training a machine learning model. And determine what factors are used to predict if the first stage will land successfully.

### 💻 Tools and Technologies:
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) 
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) 
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white) 
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23F7EBE2?style=for-the-badge&logo=pandas&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![JupyterNotebook](https://img.shields.io/badge/JupyterNotebook-FACCA7?style=for-the-badge&logo=Jupyter)
![GIT](https://img.shields.io/badge/Git-fc6d26?style=for-the-badge&logo=git&logoColor=white)
![seaborn](https://img.shields.io/badge/seaborn-86B4AD?style=for-the-badge)
![beautifulsoup](https://img.shields.io/badge/BeautifulSoup-404847?style=for-the-badge)
![Folium](https://img.shields.io/badge/Folium-F1F9E1?style=for-the-badge&logo=Folium)

### Data Sets:
Wikipedia: "https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922“
 
SpaceX JSON: 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/API_call_spacex_api.json’

### Data collection methodology 
-Data collection is process of gathering and measuring information on targeted variables in an established system, which then enables one to answer relevant questions and evaluate outcome.

-For this project the data was collected by REST API and web scrapping from Wikipedia.

![image](https://github.com/my3amarnath/SpaceX-Falcon-9-first-stage-Landing-Prediction/assets/39696237/6d71adaf-4281-4e99-978c-6cef9500ff54)

#### Data Collection – with SpaceX REST API

![image](https://github.com/my3amarnath/SpaceX-Falcon-9-first-stage-Landing-Prediction/assets/39696237/27814e7e-a6d0-403c-8374-0f97b36ad8fd)

####  Data Collection - Web Scrapping from Wikipedia 

![image](https://github.com/my3amarnath/SpaceX-Falcon-9-first-stage-Landing-Prediction/assets/39696237/8f207756-ca26-458d-91fb-e49e0318b845)

### Data Wrangling
Exploratory Data Analysis (EDA) to find some patterns in the data and determine what would be the label for training supervised models.

![image](https://github.com/my3amarnath/SpaceX-Falcon-9-first-stage-Landing-Prediction/assets/39696237/4c5b5607-1d55-48b7-986f-cb601c57711b)

### Perform exploratory data analysis (EDA) using visualization 

We used scatter plot and bar chart to find the relationship between different attributes as below.
* Use scatter plot to visualize the relationship between Flight Number and Launch Site. ![image](https://github.com/my3amarnath/SpaceX-Falcon-9-first-stage-Landing-Prediction/assets/39696237/a1cf0370-42cf-4a49-a6a2-663c7450997c)
* Use scatter plot to visualize the relationship between Payload and Launch Site.![image](https://github.com/my3amarnath/SpaceX-Falcon-9-first-stage-Landing-Prediction/assets/39696237/89028db7-9a84-4e4d-9a50-3f71784b6fb8)
* Use bar chart to visualize the relationship between success rate of each orbit type.![image](https://github.com/my3amarnath/SpaceX-Falcon-9-first-stage-Landing-Prediction/assets/39696237/6c651b2b-9f67-4676-9509-80bf7eb7764c)
* Use scatter plot to visualize the relationship between Flight Number and Orbit type.![image](https://github.com/my3amarnath/SpaceX-Falcon-9-first-stage-Landing-Prediction/assets/39696237/4c35d754-b127-4f5f-8bc6-cf349bccaaf1)
* Use line plot to visualize the launch success yearly trend.![image](https://github.com/my3amarnath/SpaceX-Falcon-9-first-stage-Landing-Prediction/assets/39696237/3286aa07-cd8f-4c97-9d5f-796c5c7830e4)

Once a pattern is determined through visualization it would be easy to understand the factors affecting the success rate of Falcon 9 landing.

### Perform exploratory data analysis (EDA) using SQL

* Display the names of the unique launch sites in the space mission.
* Display 5 records where launch sites begin with the string 'CCA’.
* Display the total payload mass carried by boosters launched by NASA (CRS).
* Display average payload mass carried by booster version F9 v1.1.
* List the date when the first successful landing outcome in ground pad was achieved.
* List the names of the boosters which have success in drone ship and have payload mass greater than 4000 but less than 6000.
* List the names of the booster versions which have carried the maximum payload mass. Use a subquery
* List the records which will display the month names, failure landing outcomes in drone ship ,booster versions, launch site for the months in year 2015.
* Rank the count of landing outcomes (such as Failure (drone ship) or Success (ground pad)) between the date 2010-06-04 and 2017-03-20, in descending order.

### Build an Interactive Map with Folium

Markers were added for launch sites and for NASA Johnson Space Center.
![image](https://github.com/my3amarnath/SpaceX-Falcon-9-first-stage-Landing-Prediction/assets/39696237/b1d24f7b-bb46-4214-8f43-09f764384487)
Circles were added for the launch sites with thecolor-labeled markers in marker clusters, you should be able to easily identify which launch sites have relatively high success rates.
![image](https://github.com/my3amarnath/SpaceX-Falcon-9-first-stage-Landing-Prediction/assets/39696237/ce919314-9e9c-41e4-9113-f8b59ddd1285)
Lines were added to explore and analyze the proximities of launch sites.
* Are launch sites in close proximity to railways?.
* Do launch sites keep certain distance away from cities?
* Are launch sites in close proximity to highways?
* Are launch sites in close proximity to coastline?
![image](https://github.com/my3amarnath/SpaceX-Falcon-9-first-stage-Landing-Prediction/assets/39696237/6a3b04c0-a6c9-4992-96b1-f5dc668dd619)


### Build a Dashboard with Plotly Dash

Dashboard was built with Plotly Dash which would allow users to derive and visualize the results they want.![Screenshot 2024-04-12 131259](https://github.com/my3amarnath/SpaceX-Falcon-9-first-stage-Landing-Prediction/assets/39696237/d9ce0e29-6fcb-4b60-853a-1a4b021d0a14)

Plotted Pie charts showing the outcome of launches for all sites and can be filtered based on launch sites by using the drop down menu.![image](https://github.com/my3amarnath/SpaceX-Falcon-9-first-stage-Landing-Prediction/assets/39696237/80a86694-ac6b-462e-b3f7-240804ee07e2) ![image](https://github.com/my3amarnath/SpaceX-Falcon-9-first-stage-Landing-Prediction/assets/39696237/4f539e70-b351-4978-a1b7-437c55115d2e)

Plotted Scatter graph showing the relationship between the outcome of launches and Payload Mass for different booster versions by using Slider to select the payload range.
![image](https://github.com/my3amarnath/SpaceX-Falcon-9-first-stage-Landing-Prediction/assets/39696237/3e0d9f70-7076-40e2-9c05-194d39239905)

### Predictive Analysis (Classification)
Perform predictive analysis using classification models by Standardize the data Split into training data and test data we find best Hyperparameter for SVM, Classification Trees and Logistic Regression and also find the method that performs best using test data.

![image](https://github.com/my3amarnath/SpaceX-Falcon-9-first-stage-Landing-Prediction/assets/39696237/885e0d62-6a7e-4ec7-9235-35da26b7173d)

#### 1) Classification Accuracy

Best Algorithm with training data is  Decision Tree with a score of 0.8625. Whereas ss per above Accuracy chart for test data - All methods perform equally on the test data and they all have same accuracy of 0.833333 on the test Data.
![image](https://github.com/my3amarnath/SpaceX-Falcon-9-first-stage-Landing-Prediction/assets/39696237/31ddde52-f8a5-4c19-b44f-20c79e0ff77d)

#### 2) Confusion Matrix
Confusion matrix can be read as 

![image](https://github.com/my3amarnath/SpaceX-Falcon-9-first-stage-Landing-Prediction/assets/39696237/c7745554-e562-40a9-af3b-5e42b24800d8)

Prediction Breakdown:
* 12 True Positives and 3 True Negatives
* 3 False Positives and 0 False Negatives

![image](https://github.com/my3amarnath/SpaceX-Falcon-9-first-stage-Landing-Prediction/assets/39696237/ce09d38c-9b0f-4a80-acc3-e47ae1bf5a99)

The chart shown is the confusion matrix for Decision Tree.



































