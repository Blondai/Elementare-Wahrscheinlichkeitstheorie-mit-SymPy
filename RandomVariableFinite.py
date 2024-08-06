import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

from .RandomVariable import RandomVariable


class RandomVariableFinite(RandomVariable):
    def __init__(self, density, variables, force_density=False):
        """
        A subclass of RandomVariable for finite random variables.

        Parameters:
            density (dict): The density function of the random variable as a dictionary mapping values to probabilities.
            variables (sympy.Symbol): The symbol used in the density function as a sympy object.
            force_density (bool): An Argument for skipping the check if the density function is normalized. Default is False.
        """
        self.type = 'f'
        density = RandomVariableFinite._make_density(density)
        # Initiates Random Variable
        super(RandomVariableFinite, self).__init__(density, variables, force_density=force_density)

    @staticmethod
    def _make_density(old_density):
        """
        Converts the density function of the random variable to a sympy expression if necessary.

        Parameters:
            old_density (dict): The density function of the random variable as a dictionary mapping values to probabilities.

        Returns:
            dict: The density function of the random variable as a dictionary mapping sympy expressions to sympy expressions.
        """
        density = {}
        for key in old_density.keys():
            density.update({sym.sympify(key): sym.sympify(old_density[key])})
        return density

    def integrate_random_variable(self, expr, lower=-sym.oo, upper=sym.oo):
        """
        Returns the sum of an expression involving the random variable.

        Parameters:
            expr (sympy.Expr): The expression to be summed as a symbolic expression.
            lower (sympy.Expr): The lower bound of the summation. Default is -sym.oo.
            upper (sympy.Expr): The upper bound of the summation. Default is sym.oo.

        Returns:
            sympy.Expr: The sum as a symbolic expression.
        """
        integral = sym.Integer(0)
        for key in self.density.keys():
            # Only values inside intervall
            if key >= lower and key <= upper:
                integral += expr.subs(self.variable, key) * self.density[key]
        integral = sym.simplify(integral)
        return integral

    def distribution_function(self):
        """
        Returns the distribution function of the random variable.

        Returns:
            dict: The distribution function of the random variable as a dictionary mapping values to cumulative probabilities.
        """
        sortable = True
        keys = list(self.density.keys())
        for key in keys:
            if isinstance(key, sym.Symbol):
                print('WARNING: Can\'t sort values.')
                sortable = False
                break
        if sortable:
            keys = sorted(keys)
        cumulative_probability = self.density[keys[0]]
        distribution_function = {keys[0]: cumulative_probability}
        keys.pop(0)
        for key in keys:
            cumulative_probability += self.density[key]
            distribution_function.update({key: cumulative_probability})
        return distribution_function

    def entropy(self):
        """
        Returns the entropy of the random variable.

        Returns:
            sympy.Expr: The entropy as a symbolic expression.
        """
        entropy = sym.Integer(0)
        for probability in self.density.values():
            entropy += probability * sym.log(probability)
        entropy = - entropy
        entropy = sym.simplify(entropy)
        return entropy

    def plot_density(self, show=True, use_latex=True):
        """
        Plots the density function of the random variable.

        Parameters:
            show (bool): A flag to indicate whether to show the plot or return the figure and axis objects. Default is True.
            use_latex (bool): A flag to indicate whether to use LaTeX for rendering the plot labels. Default is True.

        Returns:
            matplotlib.figure.Figure, matplotlib.axes.Axes: The figure and axis objects of the plot, if show is False.
        """
        if self._test_for_symbols():
            return
        x_values = list(self.density.keys())
        y_values = list(self.density.values())
        if use_latex:
            plt.rc('text', usetex=True)
        fig, ax = plt.subplots()
        ax.set_xlabel(f'${sym.latex(self.variable)}$')
        ax.set_ylabel('Density function')
        ax.scatter(x_values, y_values, marker='.')
        if show:
            plt.show()
        else:
            return fig, ax

    def plot_distribution_function(self, show=True, use_latex=True):
        """
        Plots the distribution function of the random variable.

        Parameters:
            lower (float or None): The lower bound of the plot domain. Default is None.
            upper (float or None): The upper bound of the plot domain. Default is None.
            show (bool): A flag to indicate whether to show the plot or return the figure and axis objects. Default is True.
            use_latex (bool): A flag to indicate whether to use LaTeX for rendering the plot labels. Default is False.

        Returns:
            matplotlib.figure.Figure, matplotlib.axes.Axes: The figure and axis objects of the plot, if show is False.
        """
        if self._test_for_symbols():
            return
        distribution_function = self.distribution_function()
        keys = list(distribution_function.keys())
        min_value = min(keys)
        max_value = max(keys)
        distance = max_value - min_value
        # Broader Plot
        lower = min_value - 0.1 * distance
        upper = max_value + 0.1 * distance
        if use_latex:
            plt.rc('text', usetex=True)
        fig, ax = plt.subplots()
        ax.set_xlabel(f'${sym.latex(self.variable)}$')
        ax.set_ylabel('Distribution function')
        # 0 from - infinity to first value
        ax.hlines(y=0, xmin=lower, xmax=min_value, color='tab:blue', linewidth=2)
        # 1 from last value to infinity
        ax.hlines(y=1, xmin=max_value, xmax=upper, color='tab:blue', linewidth=2)
        for num, key in enumerate(keys):
            # Skips last line
            if num < len(keys) - 1:
                ax.hlines(y=distribution_function[key], xmin=keys[num], xmax=keys[num+1], color='tab:blue', linewidth=2)
        if show:
            plt.show()
        else:
            return fig, ax

    def simulate(self, number):
        """
        Simulates the random variable using inverse transform sampling.

        Parameters:
            number (int): The number of samples to generate.

        Returns:
            list: A list of simulated values of the random variable.
        """
        simulate = []
        uni = np.random.uniform(0, 1, number)
        distribution_function = self.distribution_function()
        for num in uni:
            for key in distribution_function.keys():
                if num <= float(distribution_function[key]):
                    simulate.append(float(key))
                    break
        return simulate