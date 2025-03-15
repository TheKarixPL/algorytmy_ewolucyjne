import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms

# Problem: Optymalizacja układu reklam na stronie internetowej
# Cel: Maksymalizacja przychodów z reklam i minimalizacja dyskomfortu użytkowników

# Parametry problemu
NUM_ADS = 5  # Liczba reklam
PAGE_AREAS = 10  # Liczba dostępnych obszarów na stronie

# Funkcja fitness
def evaluate(individual):
    """
    Evaluate the ad layout:
    - Maximize ad revenue.
    - Minimize user discomfort.
    """
    ad_revenues = [10, 20, 15, 5, 25]
    user_discomfort = [3, 2, 4, 5, 1, 3, 2, 4, 5, 1]

    revenue = sum(ad_revenues[ad] for ad in range(len(individual)))
    discomfort = sum(user_discomfort[position] for position in individual)

    return revenue - discomfort,

# Inicjalizacja DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Reprezentacja układu: każda reklama przypisana do obszaru
toolbox.register("attr_area", random.randint, 0, PAGE_AREAS - 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_area, n=NUM_ADS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=PAGE_AREAS - 1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Funkcja eksperymentu z różnymi parametrami
def run_experiment(pop_size, gens, cxpb, mutpb, runs=10):
    results = []

    for run in range(runs):
        random.seed(run)
        population = toolbox.population(n=pop_size)

        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("avg", lambda x: sum(x) / len(x))
        stats.register("min", min)
        stats.register("max", max)

        population, log = algorithms.eaSimple(
            population, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=gens, stats=stats, verbose=False
        )

        # Zapisanie logów i najlepszego osobnika
        best_ind = tools.selBest(population, k=1)[0]
        results.append({
            "log": log,
            "best_ind": best_ind,
            "best_fitness": best_ind.fitness.values[0]
        })

    return results

# Parametry eksperymentów
EXPERIMENTS = [
    {"pop_size": 50, "gens": 40, "cxpb": 0.7, "mutpb": 0.2},
    {"pop_size": 100, "gens": 50, "cxpb": 0.8, "mutpb": 0.1},
    {"pop_size": 30, "gens": 30, "cxpb": 0.6, "mutpb": 0.3},
]

# Wykresy wyników
def plot_results(experiment_results, experiment_params):
    avg_fitness = []
    min_fitness = []
    max_fitness = []

    for log_entry in experiment_results[0]["log"]:
        avg_fitness.append(log_entry["avg"])
        min_fitness.append(log_entry["min"])
        max_fitness.append(log_entry["max"])

    plt.figure(figsize=(10, 6))
    plt.plot(avg_fitness, label="Avg Fitness", color="blue")
    plt.plot(min_fitness, label="Min Fitness", color="red")
    plt.plot(max_fitness, label="Max Fitness", color="green")

    plt.fill_between(
        range(len(avg_fitness)),
        [a - np.std([r["best_fitness"] for r in experiment_results]) for a in avg_fitness],
        [a + np.std([r["best_fitness"] for r in experiment_results]) for a in avg_fitness],
        color="blue",
        alpha=0.2
    )

    plt.title(f"Fitness Progression\nParams: Pop={experiment_params['pop_size']}, Gens={experiment_params['gens']}, CxPB={experiment_params['cxpb']}, MutPB={experiment_params['mutpb']}")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()

# Główne wywołanie
if __name__ == "__main__":
    for i, params in enumerate(EXPERIMENTS):
        print(f"\nRunning Experiment {i+1} with params: {params}")
        results = run_experiment(**params)

        # Omówienie najlepszego rozwiązania
        best_run = max(results, key=lambda x: x["best_fitness"])
        print(f"Best layout: {best_run['best_ind']}")
        print(f"Best fitness: {best_run['best_fitness']}")

        # Wykresy
        plot_results(results, params)
