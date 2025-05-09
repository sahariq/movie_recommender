# models/ga_optimizer.py

import pandas as pd
import random

# Load movie data
df = pd.read_csv('data/tmdb_5000_movies.csv')

# Clean and filter necessary columns
df = df[['title', 'popularity', 'vote_average', 'runtime']]
df = df.dropna()
df = df.reset_index(drop=True)

# Save max runtime for scaling back
max_runtime = df['runtime'].max()

# Normalize values for fair comparison
df['popularity'] = df['popularity'] / df['popularity'].max()
df['vote_average'] = df['vote_average'] / 10
df['runtime'] = df['runtime'] / max_runtime

# Genetic Algorithm parameters
POP_SIZE = 20
NUM_GENERATIONS = 15
CHROMOSOME_LENGTH = 5
MUTATION_RATE = 0.2

def fitness(chromosome):
    selected = df.iloc[chromosome]
    pop = selected['popularity'].mean()
    vote = selected['vote_average'].mean()
    runtime = selected['runtime'].mean()
    return (0.4 * pop) + (0.4 * vote) + (0.2 * runtime)

def generate_chromosome():
    return random.sample(range(len(df)), CHROMOSOME_LENGTH)

def crossover(parent1, parent2):
    cut = random.randint(1, CHROMOSOME_LENGTH - 1)
    child = parent1[:cut] + [gene for gene in parent2 if gene not in parent1[:cut]]
    while len(child) < CHROMOSOME_LENGTH:
        gene = random.randint(0, len(df) - 1)
        if gene not in child:
            child.append(gene)
    return child

def mutate(chromosome):
    if random.random() < MUTATION_RATE:
        i = random.randint(0, CHROMOSOME_LENGTH - 1)
        gene = random.randint(0, len(df) - 1)
        while gene in chromosome:
            gene = random.randint(0, len(df) - 1)
        chromosome[i] = gene
    return chromosome

def genetic_algorithm():
    population = [generate_chromosome() for _ in range(POP_SIZE)]
    for generation in range(NUM_GENERATIONS):
        population = sorted(population, key=fitness, reverse=True)
        next_gen = population[:2]  # Elitism
        while len(next_gen) < POP_SIZE:
            parent1, parent2 = random.sample(population[:10], 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            next_gen.append(child)
        population = next_gen
        print(f"Generation {generation+1} Best Fitness: {fitness(population[0]):.4f}")

    best = population[0]
    return df.iloc[best]
