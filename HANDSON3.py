class PolynomialLinearRegression:
    def __init__(self, dataset):
        self.dataset = dataset
        self.x_values = dataset['x']
        self.y_values = dataset['y']

    def linear_regression(self):
        n = len(self.dataset)
        sum_x = sum(self.x_values)
        sum_y = sum(self.y_values)
        sum_x_squared = sum(x ** 2 for x in self.x_values)
        sum_xy = sum(x * y for x, y in zip(self.x_values, self.y_values))

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
        intercept = (sum_y - slope * sum_x) / n

        return slope, intercept

    def quadratic_regression(self):
        import numpy as np
        x_squared = np.square(self.x_values)
        xy = np.multiply(self.x_values, self.y_values)
        x_cubed = np.power(self.x_values, 3)

        A = np.vstack([x_squared, self.x_values, np.ones(len(self.x_values))]).T
        coeffs = np.linalg.lstsq(A, self.y_values, rcond=None)[0]

        return coeffs

    def cubic_regression(self):
        import numpy as np
        x_squared = np.square(self.x_values)
        x_cubed = np.power(self.x_values, 3)
        x_quad = np.power(self.x_values, 4)
        xy = np.multiply(self.x_values, self.y_values)
        x_squared_y = np.multiply(x_squared, self.y_values)

        A = np.vstack([x_quad, x_cubed, x_squared, self.x_values, np.ones(len(self.x_values))]).T
        coeffs = np.linalg.lstsq(A, self.y_values, rcond=None)[0]

        return coeffs

    def predict(self, x_value, coeffs):
        prediction = 0
        for i, coeff in enumerate(reversed(coeffs)):
            prediction += coeff * (x_value ** i)
        return prediction

    def correlation_and_determination(self, y_values, predicted_y_values):
        import numpy as np
        correlation = np.corrcoef(y_values, predicted_y_values)[0, 1]
        determination = correlation ** 2
        return correlation, determination

    def print_equation(self, coeffs, degree):
        equation = "y = "
        for i, coeff in enumerate(reversed(coeffs)):
            if i < degree:
                equation += f"{coeff}x^{degree - i} + "
            elif i == degree:
                equation += f"{coeff}"
        print(f"Equación de Regresión Polinomial de grado {degree}: {equation}")

    def run(self):
        linear_coeffs = self.linear_regression()
        quadratic_coeffs = self.quadratic_regression()
        cubic_coeffs = self.cubic_regression()

        self.print_equation(linear_coeffs, 1)
        self.print_equation(quadratic_coeffs, 2)
        self.print_equation(cubic_coeffs, 3)

        # Predicciones
        known_values = [self.x_values[0], self.x_values[int(len(self.x_values) / 2)], self.x_values[-1]]
        for value in known_values:
            linear_prediction = self.predict(value, linear_coeffs)
            quadratic_prediction = self.predict(value, quadratic_coeffs)
            cubic_prediction = self.predict(value, cubic_coeffs)
            print(f"Predicciones para x = {value}:")
            print(f"  - Lineal: {linear_prediction}")
            print(f"  - Cuadrática: {quadratic_prediction}")
            print(f"  - Cúbica: {cubic_prediction}")

        # Coeficientes de correlación y determinación
        linear_predicted_y = [self.predict(x, linear_coeffs) for x in self.x_values]
        quadratic_predicted_y = [self.predict(x, quadratic_coeffs) for x in self.x_values]
        cubic_predicted_y = [self.predict(x, cubic_coeffs) for x in self.x_values]

        linear_corr, linear_det = self.correlation_and_determination(self.y_values, linear_predicted_y)
        quadratic_corr, quadratic_det = self.correlation_and_determination(self.y_values, quadratic_predicted_y)
        cubic_corr, cubic_det = self.correlation_and_determination(self.y_values, cubic_predicted_y)

        print("Coeficientes de correlación y determinación:")
        print(f"  - Lineal: Correlación = {linear_corr}, Determinación = {linear_det}")
        print(f"  - Cuadrática: Correlación = {quadratic_corr}, Determinación = {quadratic_det}")
        print(f"  - Cúbica: Correlación = {cubic_corr}, Determinación = {cubic_det}")


#dataset
dataset = {'x': [108, 115, 106, 97, 95, 91, 97, 83, 83, 78, 54, 67, 56, 53, 61, 115, 81, 78, 30, 45, 99, 32, 25, 28, 90, 89],
           'y': [95, 96, 95, 97, 93, 94, 95, 93, 92, 86, 73, 80, 65, 69, 77, 96, 87, 89, 60, 63, 95, 61, 55, 56, 94, 93]}
plr = PolynomialLinearRegression(dataset)
plr.run()