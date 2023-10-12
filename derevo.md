class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        # Конструктор класса, устанавливающий параметры модели
        self.n_estimators = n_estimators      # количество деревьев в лесу
        self.max_depth = max_depth            # максимальная глубина деревьев
        self.random_state = random_state      # случайное начальное состояние для генератора случайных чисел
        self.estimators = []                  # список для хранения деревьев
    
    def fit(self, X, y):
        # Метод для обучения модели на тренировочных данных X и метках y
        rng = np.random.default_rng(self.random_state)  # инициализация генератора случайных чисел
        for i in range(self.n_estimators):
            # Использование Bootstrap Aggregating для случайного выбора объектов для текущего дерева
            idxs = rng.choice(X.shape[0], X.shape[0])
            X_subset, y_subset = X[idxs], y[idxs]
            
            # Создание и обучение дерева решений с заданными параметрами
            clf = DecisionTree(max_depth=self.max_depth)
            clf.fit(X_subset, y_subset)
            self.estimators.append(clf)  # Добавление обученного дерева в список
    
    def predict(self, X):
        # Метод для предсказания меток на новых данных X
        y_pred = []
        for i in range(len(X)):
            votes = {}  # Словарь для подсчета голосов деревьев за каждый класс
            for clf in self.estimators:
                pred = clf.predict([X[i]])[0]  # Предсказание метки на текущем дереве
                if pred not in votes:
                    votes[pred] = 1
                else:
                    votes[pred] += 1
            y_pred.append(max(votes, key=votes.get))  # Выбор метки с наибольшим количеством голосов
        return np.array(y_pred)  # Возвращение предсказанных меток в виде массива numpy


Далее обучим и оценим точность нашего алгоритма:
# Генерируем данные для обучения
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

clf = RandomForestClassifier(n_estimators=100, max_depth=3)
clf.fit(X, y)

# Прогнозируем метки классов 
y_pred = clf.predict(X)

print(classification_report(y, y_pred))

OUT:
              precision    recall  f1-score   support

           0       0.91      0.94      0.93       500
           1       0.94      0.91      0.92       500

    accuracy                           0.93      1000
   macro avg       0.93      0.93      0.92      1000
weighted avg       0.93      0.93      0.92      1000
