"""
Grey Wolf Optimizer logic.

"""

from __future__ import annotations

import numpy as np

from models import ProductionProblem, OptimizationResult


def calculate_metrics(problem: ProductionProblem, x: np.ndarray) -> dict:
    """
    Calculate production metrics for a candidate solution.
    """

    x = np.round(np.asarray(x, dtype=float))

    return {
        "profit": float(np.dot(problem.profits, x)),
        "pollution": float(np.dot(problem.pollution, x)),
        "time": float(np.dot(problem.time, x)),
        "resources": float(np.dot(problem.resources, x)),
    }


def constraint_status(problem: ProductionProblem, x: np.ndarray) -> dict:
    """
    Return whether a solution satisfies each major constraint.
    """

    x = np.round(np.asarray(x, dtype=float))
    metrics = calculate_metrics(problem, x)

    return {
        "demand": bool(np.all(x <= problem.demand) and np.all(x >= 0)),
        "time": bool(metrics["time"] <= problem.max_time),
        "resources": bool(metrics["resources"] <= problem.max_resources),
    }


def fitness(problem: ProductionProblem, x: np.ndarray) -> float:
    """
    Fitness function.

    Lower values are better.

    Objective:
    - Maximize profit.
    - Minimize pollution.
    - Penalize time, resource, and demand violations.
    """

    x = np.round(np.asarray(x, dtype=float))
    metrics = calculate_metrics(problem, x)

    score = -metrics["profit"] + problem.pollution_weight * metrics["pollution"]

    penalty = 0.0

    if metrics["time"] > problem.max_time:
        penalty += problem.constraint_penalty * (metrics["time"] - problem.max_time)

    if metrics["resources"] > problem.max_resources:
        penalty += problem.constraint_penalty * (
            metrics["resources"] - problem.max_resources
        )

    lower_violation = np.maximum(0, -x)
    upper_violation = np.maximum(0, x - problem.demand)

    penalty += problem.constraint_penalty * float(
        np.sum(lower_violation) + np.sum(upper_violation)
    )

    return float(score + penalty)


def is_problem_feasible(problem: ProductionProblem) -> bool:
    """
    Basic feasibility check.

    A fully rigorous feasibility check would require solving an integer program.
    This lightweight check verifies that at least one product can be produced
    without immediately violating the global constraints.
    """

    for i in range(problem.dimensions):
        if (
            problem.demand[i] > 0
            and problem.time[i] <= problem.max_time
            and problem.resources[i] <= problem.max_resources
        ):
            return True

    return False


def grey_wolf_optimizer(
    problem: ProductionProblem,
    num_wolves: int = 30,
    max_iter: int = 100,
    seed: int = 42,
) -> OptimizationResult:
    """
    Run Grey Wolf Optimization.

    The implementation uses the standard GWO update equations with Alpha, Beta,
    and Delta leaders. Candidate positions are clipped to demand limits.
    """

    if problem.dimensions < 1:
        raise ValueError("The problem must contain at least one product.")

    if num_wolves < 4:
        raise ValueError("Number of wolves must be at least 4.")

    if max_iter < 1:
        raise ValueError("Number of iterations must be at least 1.")

    rng = np.random.default_rng(seed)

    wolves = rng.uniform(
        low=np.zeros(problem.dimensions),
        high=problem.demand,
        size=(num_wolves, problem.dimensions),
    )

    histories = {
        "fitness": {"Alpha": []},
        "profit": {"Alpha": [], "Beta": [], "Delta": [], "Omega": []},
        "pollution": {"Alpha": [], "Beta": [], "Delta": [], "Omega": []},
        "time": {"Alpha": [], "Beta": [], "Delta": [], "Omega": []},
        "resources": {"Alpha": [], "Beta": [], "Delta": [], "Omega": []},
    }

    alpha = wolves[0].copy()
    beta = wolves[1].copy()
    delta = wolves[2].copy()
    alpha_fit = fitness(problem, alpha)

    for iteration in range(max_iter):
        fitness_values = np.array([fitness(problem, wolf) for wolf in wolves])
        indexes = np.argsort(fitness_values)

        alpha = wolves[indexes[0]].copy()
        beta = wolves[indexes[1]].copy()
        delta = wolves[indexes[2]].copy()
        omega = wolves[indexes[3:]].copy()

        alpha_fit = float(fitness_values[indexes[0]])

        rounded_leaders = {
            "Alpha": np.round(alpha),
            "Beta": np.round(beta),
            "Delta": np.round(delta),
        }

        for role, solution in rounded_leaders.items():
            metrics = calculate_metrics(problem, solution)
            histories["profit"][role].append(metrics["profit"])
            histories["pollution"][role].append(metrics["pollution"])
            histories["time"][role].append(metrics["time"])
            histories["resources"][role].append(metrics["resources"])

        if len(omega) > 0:
            omega_rounded = np.round(omega)
            omega_metrics = [
                calculate_metrics(problem, wolf) for wolf in omega_rounded
            ]

            for key in ["profit", "pollution", "time", "resources"]:
                histories[key]["Omega"].append(
                    float(np.mean([m[key] for m in omega_metrics]))
                )
        else:
            metrics = calculate_metrics(problem, np.round(alpha))
            for key in ["profit", "pollution", "time", "resources"]:
                histories[key]["Omega"].append(metrics[key])

        histories["fitness"]["Alpha"].append(alpha_fit)

        a = 2 - iteration * (2 / max_iter)

        for i in range(num_wolves):
            for j in range(problem.dimensions):
                r1, r2 = rng.random(), rng.random()
                a1 = 2 * a * r1 - a
                c1 = 2 * r2
                d_alpha = abs(c1 * alpha[j] - wolves[i, j])
                x1 = alpha[j] - a1 * d_alpha

                r1, r2 = rng.random(), rng.random()
                a2 = 2 * a * r1 - a
                c2 = 2 * r2
                d_beta = abs(c2 * beta[j] - wolves[i, j])
                x2 = beta[j] - a2 * d_beta

                r1, r2 = rng.random(), rng.random()
                a3 = 2 * a * r1 - a
                c3 = 2 * r2
                d_delta = abs(c3 * delta[j] - wolves[i, j])
                x3 = delta[j] - a3 * d_delta

                wolves[i, j] = (x1 + x2 + x3) / 3

            wolves[i] = np.clip(wolves[i], 0, problem.demand)

    best_solution = np.round(alpha)
    best_metrics = calculate_metrics(problem, best_solution)
    best_fitness = fitness(problem, best_solution)

    return OptimizationResult(
        solution=best_solution,
        profit=best_metrics["profit"],
        pollution=best_metrics["pollution"],
        time=best_metrics["time"],
        resources=best_metrics["resources"],
        fitness=best_fitness,
        constraint_status=constraint_status(problem, best_solution),
        histories=histories,
    )
