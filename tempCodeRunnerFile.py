start = time.time()
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(random_state= 42)
log_reg.fit(x_train, y_train)
y_pred = log_reg.predict(x_test)
end = time.time()