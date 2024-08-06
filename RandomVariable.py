import sympy as sym
import numpy as np
import matplotlib.pyplot as plt


class RandomVariable:
    CLEAN_PIECEWISE = True

    def __init__(self, density, variable, force_density=False):
        """
        Initializes a random variable object with a given density function and symbol used in the density function.

        Parameters:
            density (sympy.Expr or dict): The density function of the random variable as a symbolic expression or dictionary.
            variable (sympy.Symbol): The symbol used in the density function as a sympy object.
            force_density (bool): An Argument for skipping the check if the density function is normalized. Default is False.
        """
        self.density = density
        self.variable = variable
        self.force_density = force_density
        if force_density == False:
            self._is_density()

    def integrate_random_variable(self):
        pass

    @staticmethod
    def clean_piecewise(expr):
        """
        Cleans up a piecewise expression by removing unnecessary branches and arguments.

        Parameters:
            expr (sympy.Expr): The expression to be cleaned up.

        Returns:
            sympy.Expr: The cleaned up expression.
        """
        if not RandomVariable.CLEAN_PIECEWISE:
            return expr
        if isinstance(expr, sym.Piecewise):
            print('WARNING: Chopped up piecewise-function')
            expr = expr.args[0][0]
        elif expr.is_Atom:
            expr = expr
        # Repeats previous steps for all parts
        else:
            args = [RandomVariable.clean_piecewise(arg) for arg in expr.args]
            expr = expr.func(*args)
        return expr

    @staticmethod
    def no_chopping():
        """
        Disables the clean_piecewise method.
        """
        RandomVariable.CLEAN_PIECEWISE = False

    def _test_for_symbols(self):
        """
        Tests if the density function of the random variable has multiple symbols.

        Returns:
            bool: True if the density function has multiple symbols, False otherwise.
        """
        if self.type == 'f':
            # Tests key for symbols
            for key in self.density.keys():
                if isinstance(key, sym.Symbol):
                    print('ERROR: Multiple symbols in density not supported.')
                    return True
            # Tests values for symbols
            for value in self.density.values():
                if isinstance(value, sym.Symbol):
                    print('ERROR: Multiple symbols in density not supported.')
                    return True
            return False
        else:
            if set(self.density.free_symbols) != set([self.variable]):
                print('ERROR: Multiple symbols in density not supported.')
                return True
            else:
                return False

    def _is_density(self):
        """
        Checks if the density function of the random variable is normalized.

        Returns:
            sympy.Expr: The integral of the density function over the whole domain.
        """
        total = self.integrate_random_variable(sym.Integer(1))
        if not total.equals(sym.Integer(1)):
            print('WARNING: Density not standardized!')
        return total

    # Methods for raw moments
    def moment_generating_function(self):
        """
        Returns the moment generating function of the random variable.

        Returns:
            sympy.Expr: The moment generating function as a symbolic expression.
        """
        # Tests for moment generating function attribute
        if hasattr(self, 'MGF'):
            moment_generating_function = self.MGF
        else:
            t = sym.Symbol('t', real=True)
            moment_generating_function = self.integrate_random_variable(sym.exp(t * self.variable))
            # Sets moment generating function attribute
            self.MGF = moment_generating_function
        return moment_generating_function

    def _moment_integration(self, n):
        """
        Returns the nth raw moment of the random variable using integration.

        Parameters:
            n (int): The order of the moment.

        Returns:
            sympy.Expr: The nth raw moment as a symbolic expression.
        """
        moment = self.integrate_random_variable(self.variable**n)
        return moment

    def _moment_generating(self, n):
        """
        Returns the nth raw moment of the random variable using differentiation of the moment generating function.

        Parameters:
            n (int): The order of the moment.

        Returns:
            sympy.Expr: The nth raw moment as a symbolic expression.
        """
        t = sym.Symbol('t', real=True)
        moment_generating_function = self.moment_generating_function()
        moment = sym.diff(moment_generating_function, (t, n))
        moment = moment.subs(t, sym.Integer(0))
        moment = sym.simplify(moment)
        return moment

    def moment(self, n, use_integration=True):
        """
        Returns the nth raw moment of the random variable.

        Parameters:
            n (int): The order of the moment.
            use_integration (bool): A flag to indicate whether to use integration or moment generating function to calculate the raw moment. Default is True.

        Returns:
            sympy.Expr: The nth raw moment as a symbolic expression.
        """
        if use_integration == True:
            moment = self._moment_integration(n)
        else:
            moment = self._moment_generating(n)
        return moment

    # Methods for central moments
    def central_moment_generating_function(self):
        """
        Returns the central moment generating function of the random variable.

        Returns:
            sympy.Expr: The central moment generating function as a symbolic expression.
        """
        # Tests for central moment generating function attribute
        if hasattr(self, 'CMGF'):
            central_moment_generating_function = self.CMGF
        else:
            t = sym.Symbol('t', real=True)
            mu = self.mean()
            moment_generating_function = self.moment_generating_function()
            central_moment_generating_function = sym.exp(- mu * t) * moment_generating_function
            central_moment_generating_function = sym.simplify(central_moment_generating_function)
            # Sets central moment generating function attribute
            self.CMGF = central_moment_generating_function
        return central_moment_generating_function

    def _central_moment_integration(self, n):
        """
        Returns the nth central moment of the random variable by integration.

        Parameters:
            n (int): The order of the central moment.

        Returns:
            sympy.Expr: The nth central moment as a symbolic expression.
        """
        mean = self.mean()
        central_moment = self.integrate_random_variable((self.variable - mean)**n)
        return central_moment

    def _central_moment_generating(self, n):
        """
        Returns the nth central moment of the random variable by using differentiation of the moment generating function.

        Parameters:
            n (int): The order of the central moment.

        Returns:
            sympy.Expr: The nth central moment as a symbolic expression.
        """
        t = sym.Symbol('t', real=True)
        central_moment_generating_function = self.central_moment_generating_function()
        central_moment = sym.diff(central_moment_generating_function, (t, n))
        central_moment = central_moment.subs(t, sym.Integer(0))
        central_moment = sym.simplify(central_moment)
        return central_moment

    def central_moment(self, n, use_integration=True):
        """
        Returns the nth central moment of the random variable.

        Parameters:
            n (int): The order of the central moment.
            use_integration (bool): A flag to indicate whether to use integration or moment generating function to calculate the central moment. Default is True.

        Returns:
            sympy.Expr: The nth central moment as a symbolic expression.
        """
        if use_integration == True:
            central_moment = self._central_moment_integration(n)
        else:
            central_moment = self._central_moment_generating(n)
        return central_moment

    # Methods for standardized moments
    def standardized_moment_generating_function(self):
        """
        Returns the standardized moment generating function of the random variable.

        Returns:
            sympy.Expr: The standardized moment generating function as a symbolic expression.
        """
        # Tests for standardized moment generating function attribute
        if hasattr(self, 'SMGF'):
            standardized_moment_generating_function = self.SMGF
        else:
            t = sym.Symbol('t', real=True)
            mu = self.mean()
            sigma = self.standard_deviation()
            moment_generating_function = self.moment_generating_function().subs(t, t / sigma)
            standardized_moment_generating_function = sym.exp(- mu / sigma * t) * moment_generating_function
            standardized_moment_generating_function = sym.simplify(standardized_moment_generating_function)
            # Sets standardized moment generating function attribute
            self.SMGF = standardized_moment_generating_function
        return standardized_moment_generating_function

    def _standard_moment_integration(self, n):
        """
        Returns the nth standardized moment of the random variable by integration.

        Parameters:
            n (int): The order of the standardized moment.

        Returns:
            sympy.Expr: The nth standardized moment as a symbolic expression.
        """
        central_moment = self.central_moment(n, use_integration=True)
        standard_deviation = self.standard_deviation(use_integration=True)
        standard_moment = central_moment / standard_deviation**n
        standard_moment = sym.simplify(standard_moment)
        return standard_moment

    def _standard_moment_generating(self, n):
        """
        Returns the nth standardized moment of the random variable by using the moment generating function.

        Parameters:
            n (int): The order of the standardized moment.

        Returns:
            sympy.Expr: The nth standardized moment as a symbolic expression.
        """
        t = sym.Symbol('t', real=True)
        standardized_moment_generating_function = self.standardized_moment_generating_function()
        standardized_moment = sym.diff(standardized_moment_generating_function, (t, n))
        standardized_moment = standardized_moment.subs(t, sym.Integer(0))
        standardized_moment = sym.simplify(standardized_moment)
        return standardized_moment

    def standard_moment(self, n, use_integration=True):
        """
        Returns the nth standardized moment of the random variable.

        Parameters:
            n (int): The order of the standardized moment.
            use_integration (bool): A flag to indicate whether to use integration or moment generating function to calculate the standardized moment. Default is True.

        Returns:
            sympy.Expr: The nth standardized moment as a symbolic expression.
        """
        if use_integration == True:
            standard_moment = self._standard_moment_integration(n)
        else:
            standard_moment = self._standard_moment_generating(n)
        return standard_moment

    def absolute_moment(self, n):
        """
        Returns the nth absolute moment of the random variable.

        Parameters:
            n (int): The order of the absolute moment.

        Returns:
            sympy.Expr: The nth absolute moment as a symbolic expression.
        """
        absolute_moment = self.integrate_random_variable(sym.Abs(self.variable)**n)
        return absolute_moment

    # Methods for cumulants
    def cumulant_generating_function(self):
        """
        Returns the cumulant generating function of the random variable.

        Returns:
            sympy.Expr: The cumulant generating function as a symbolic expression.
        """
        # Tests for cumulant generating function attribute
        if hasattr(self, 'CGF'):
            cumulant_generating_function = self.CGF
        else:
            moment_generating_function = self.moment_generating_function()
            cumulant_generating_function = sym.log(moment_generating_function)
            cumulant_generating_function = sym.simplify(cumulant_generating_function)
            # Sets cumulant generating function attribute
            self.CHF = cumulant_generating_function
        return cumulant_generating_function

    def cumulant(self, n):
        """
        Returns the nth cumulant of the random variable using differentiation of the cumulant generating function.

        Parameters:
            n (int): The order of the cumulant.

        Returns:
            sympy.Expr: The nth cumulant as a symbolic expression.
        """
        cumulant_generating_function = self.cumulant_generating_function()
        t = sym.Symbol('t', real=True)
        cumulant = sym.diff(cumulant_generating_function, t, n)
        cumulant = cumulant.subs(t, 0)
        cumulant = sym.simplify(cumulant)
        return cumulant

    # Special moments
    def mean(self, use_integration=True):
        """
        Returns the mean of the random variable.

        Parameters:
            use_integration (bool): A flag to indicate whether to use integration or moment generating function to calculate the mean. Default is True.

        Returns:
            sympy.Expr: The mean as a symbolic expression.
        """
        mean = self.moment(1, use_integration=use_integration)
        return mean

    def variance(self, use_integration=True):
        """
        Returns the variance of the random variable.

        Parameters:
            use_integration (bool): A flag to indicate whether to use integration or moment generating function to calculate the variance. Default is True.

        Returns:
            sympy.Expr: The variance as a symbolic expression.
        """
        variance = self.central_moment(2, use_integration=use_integration)
        return variance

    def standard_deviation(self, use_integration=True):
        """
        Returns the standard deviation of the random variable.

        Parameters:
            use_integration (bool): A flag to indicate whether to use integration or moment generating function to calculate the standard deviation. Default is True.

        Returns:
            sympy.Expr: The standard deviation as a symbolic expression.
        """
        variance = self.variance(use_integration=use_integration)
        standard_deviation = sym.sqrt(variance)
        standard_deviation = sym.simplify(standard_deviation)
        return standard_deviation

    def skewness(self, use_integration=True):
        """
        Returns the skewness of the random variable.

        Parameters:
            use_integration (bool): A flag to indicate whether to use integration or moment generating function to calculate the skewness. Default is True.

        Returns:
            sympy.Expr: The skewness as a symbolic expression.
        """
        skewness = self.standard_moment(3, use_integration=use_integration)
        return skewness

    def kurtosis(self, use_integration=True):
        """
        Returns the kurtosis of the random variable.

        Parameters:
            use_integration (bool): A flag to indicate whether to use integration or moment generating function to calculate the kurtosis. Default is True.

        Returns:
            sympy.Expr: The kurtosis as a symbolic expression.
        """
        kurtosis = self.standard_moment(4, use_integration=use_integration)
        return kurtosis

    def excess_kurtosis(self, use_integration=True):  # Exzess
        """
        Returns the excess kurtosis of the random variable.

        Parameters:
            use_integration (bool): A flag to indicate whether to use integration or moment generating function to calculate the excess kurtosis. Default is True.

        Returns:
            sympy.Expr: The excess kurtosis as a symbolic expression.
        """
        kurtosis = self.kurtosis(use_integration=use_integration)
        excess_kurtosis = kurtosis - sym.Integer(3)
        excess_kurtosis = sym.simplify(excess_kurtosis)
        return excess_kurtosis

    def hyperskewness(self, use_integration=True):
        """
        Returns the hyperskewness of the random variable.

        Parameters:
            use_integration (bool): A flag to indicate whether to use integration or moment generating function to calculate the hyperskewness. Default is True.

        Returns:
            sympy.Expr: The hyperskewness as a symbolic expression.
        """
        hyperskewness = self.standard_moment(5, use_integration=use_integration)
        return hyperskewness

    def hypertailedness(self, use_integration=True):
        """
        Returns the hypertailedness of the random variable.

        Parameters:
            use_integration (bool): A flag to indicate whether to use integration or moment generating function to calculate the hypertailedness. Default is True.

        Returns:
            sympy.Expr: The hypertailedness as a symbolic expression.
        """
        hypertailedness = self.standard_moment(6, use_integration=use_integration)
        return hypertailedness

    def interpret_excess_kurtosis(self, use_integration=True):  # Englische Ausgabe
        """
        Prints the interpretation of the excess kurtosis of the random variable.

        Parameters:
            use_integration (bool): A flag to indicate whether to use integration or moment generating function to calculate the excess kurtosis. Default is True.
        """
        excess_kurtosis = self.excess_kurtosis(use_integration=use_integration)
        if excess_kurtosis.free_symbols != set():
            print('ERROR: Symbols in excess kurtosis not supported.')
        if excess_kurtosis == 0:
            kind = "mesokurtic"
        elif excess_kurtosis > 0:
            kind = "leptokurtic"
        elif excess_kurtosis < 0:
            kind = "platykurtic"
        print(f"The Distribution is {kind}.")

    def coefficient_of_variation(self, use_integration=True):
        """
        Returns the coefficient of variation of the random variable.

        Parameters:
            use_integration (bool): A flag to indicate whether to use integration or moment generating function to calculate the coefficient of variation. Default is True.

        Returns:
            sympy.Expr: The coefficient of variation as a symbolic expression.
        """
        mean = self.mean(use_integration=use_integration)
        standard_deviation = self.standard_deviation(use_integration=use_integration)
        coefficient_of_variation = standard_deviation / mean
        coefficient_of_variation = sym.simplify(coefficient_of_variation)
        return coefficient_of_variation

    def entropy(self):
        """
        Returns the entropy of the random variable.

        Returns:
            sympy.Expr: The entropy as a symbolic expression.
        """
        entropy = self.integrate_random_variable(- sym.log(self.density))
        return entropy

    def mean_absolute_deviation(self):
        """
        Returns the mean absolute deviation of the random variable.

        Returns:
            sympy.Expr: The mean absolute deviation as a symbolic expression.
        """
        absolute_first_moment = self.absolute_moment(1)
        standard_deviation = self.standard_deviation()
        mean_absolute_deviation = absolute_first_moment / standard_deviation
        mean_absolute_deviation = sym.simplify(mean_absolute_deviation)
        return mean_absolute_deviation

    def characteristic_function(self):
        """
        Returns the characteristic function of the random variable.

        Returns:
            sympy.Expr: The characteristic function as a symbolic expression.
        """
        t = sym.Symbol('t', real=True)
        characteristic_function = self.integrate_random_variable(sym.exp(sym.I * t * self.variable))
        return characteristic_function

    def plot_moment_generating_function(self, lower=-1, upper=1, numpoints=100, show=True, use_latex=True):
        """
        Plots the moment generating function of the random variable.

        Parameters:
            lower (float): The lower bound of the plot domain. Default is -1.
            upper (float): The upper bound of the plot domain. Default is 1.
            numpoints (int): The number of points to plot. Default is 100.
            show (bool): A flag to indicate whether to show the plot or return the figure and axis objects. Default is True.
            use_latex (bool): A flag to indicate whether to use LaTeX for rendering the plot labels. Default is True.

        Returns:
            matplotlib.figure.Figure, matplotlib.axes.Axes: The figure and axis objects of the plot, if show is False.
        """
        if self._test_for_symbols():
            return
        t = sym.Symbol('t', real=True)
        moment_generating_function = self.moment_generating_function()
        x_values = np.linspace(lower, upper, num=numpoints)
        y_values = []
        for x_value in x_values:
            y_value = float(moment_generating_function.subs(t, x_value).evalf())
            y_values.append(y_value)
        if use_latex:
            plt.rc('text', usetex=True)
        fig, ax = plt.subplots()
        ax.set_xlabel(f'${sym.latex(t)}$')
        ax.set_ylabel('Momentgenerating function')
        ax.plot(x_values, y_values)
        if show:
            plt.show()
        else:
            return fig, ax
