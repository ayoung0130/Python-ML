#!/usr/bin/env python
# coding: utf-8

# # 1. Linear Regression
# ### 공부 시간에 따른 시험 점수

# In[2]:


import matplotlib.pyplot as plt
import pandas as pd


# In[3]:


dataset = pd.read_csv('LinearRegressionData.csv')


# In[4]:


dataset.head()


# In[5]:


X = dataset.iloc[:, :-1].values # 처음부터 마지막 컬럼 직전까지의 데이터(독립 변수-원인)
y = dataset.iloc[:, -1].values # 마지막 컬럼 데이터(종속 변수-결과)


# In[6]:


X, y


# In[7]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression() # 객체 생성
reg.fit(X, y) # 학습(모델 생성)


# In[8]:


y_pred = reg.predict(X) # X에 대한 예측 값
y_pred


# In[10]:


plt.scatter(X, y, color = 'blue') # 산점도
plt.plot(X, y_pred, color = 'green') # 선 그래프
plt.title('Score by hours') # 제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') # Y 축 이름
plt.show()


# In[13]:


print('9시간 공부했을 때 예상 점수: ', reg.predict([[9]]))


# In[14]:


reg.coef_ # 기울기(m)


# In[15]:


reg.intercept_ # y 절편(b)


# y = mx + b -> y = 10.4436x - 0.2184

# ### 데이터 세트 분리

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


dataset = pd.read_csv('LinearRegressionData.csv')
dataset


# In[3]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[4]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) #훈련 80: 테스트 20 으로 분리 


# In[5]:


X, len(X) # 전체 데이터 X, 개수


# In[6]:


X_train, len(X_train) # 훈련 세트 X,  개수


# In[7]:


X_test, len(X_test) # 테스트 세트 X, 개수


# In[8]:


y, len(y) # 전체 데이터 y


# In[9]:


y_train, len(y_train) # 훈련 세트 y,  개수


# In[10]:


y_test, len(y_test) # 테스트 세트 y, 개수


# ### 분리된 데이터를 통한 모델링

# In[11]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()


# In[12]:


reg.fit(X_train, y_train) # 훈련 세트로 학습


# ### 데이터 시각화(훈련 세트)

# In[16]:


plt.scatter(X_train, y_train, color = 'blue') # 산점도
plt.plot(X_train, reg.predict(X_train), color = 'green') # 선 그래프
plt.title('Score by hours(train data)') # 제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') # Y 축 이름
plt.show()


# ### 데이터 시각화(테스트 세트)

# In[17]:


plt.scatter(X_test, y_test, color = 'blue') # 산점도
plt.plot(X_train, reg.predict(X_train), color = 'green') # 선 그래프
plt.title('Score by hours(test data)') # 제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') # Y 축 이름
plt.show()


# In[18]:


reg.coef_


# In[19]:


reg.intercept_


# ### 모델 평가

# In[20]:


reg.score(X_test, y_test) # 테스트 세트를 통한 모델 평가. 1과 가까울수록 good


# In[21]:


reg.score(X_train, y_train) # 훈련 세트를 통한 모델 평가


# ### 경사 하강법(Gradient Descent)

# max_iter : 훈련 세트 반복 횟수 (Epoch 횟수)
# 
# eta0 : 학습률 (learning rate)

# In[40]:


from sklearn.linear_model import SGDRegressor # SGD: Stochastic Gradient Descent 확률적 경사 하강법
#sr = SGDRegressor(max_iter=1000, eta0=1e-4, random_state=0, verbose=1)
sr = SGDRegressor()
sr.fit(X_train, y_train)


# In[41]:


plt.scatter(X_train, y_train, color = 'blue') # 산점도
plt.plot(X_train, sr.predict(X_train), color = 'green') # 선 그래프
plt.title('Score by hours(train data, SGD)') # 제목
plt.xlabel('hours') # X 축 이름
plt.ylabel('score') # Y 축 이름
plt.show()


# In[26]:


sr.coef_, sr.intercept_


# In[27]:


sr.score(X_test, y_test) # 테스트 세트를 통한 모델 평가


# In[28]:


sr.score(X_train, y_train) # 훈련 세트를 통한 모델 평가

