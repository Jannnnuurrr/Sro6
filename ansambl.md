Импортируйте необходимые библиотеки:
python
Copy code
from sklearn.ensemble import RandomForestClassifier  # Пример с использованием случайного леса
from sklearn.ensemble import GradientBoostingRegressor  # Пример с использованием градиентного бустинга
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
Загрузите и подготовьте данные:
python
Copy code
# Загрузите данные и разделите их на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
Создайте и обучите ансамбль:
python
Copy code
# Пример создания и обучения случайного леса
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Пример создания и обучения градиентного бустинга
gb_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_regressor.fit(X_train, y_train)
Сделайте предсказания:
python
Copy code
# Получите предсказания от ансамблей
rf_predictions = rf_classifier.predict(X_test)
gb_predictions = gb_regressor.predict(X_test)
Оцените производительность ансамблей:
python
Copy code
# Пример оценки для классификации
accuracy_rf = accuracy_score(y_test, rf_predictions)

# Пример оценки для регрессии
mse_gb = mean_squared_error(y_test, gb_predictions)
