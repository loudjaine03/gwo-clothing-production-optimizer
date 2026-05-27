# GWO Clothing Production Optimizer

A desktop application that uses the **Grey Wolf Optimizer (GWO)** to help plan clothing production.

The user enters product data, production constraints, and algorithm settings. The application then recommends the best production quantities while balancing profit, pollution, time, resources, and demand limits.

---

## Objectives

The main objectives of this project are:

- Maximize total profit
- Minimize production time
- Minimize resource usage
- Minimize environmental impact

- Respect product demand limits
- Provide clear production recommendations through a graphical interface

---

## Project Concept

In a clothing factory, the manager must decide how many units of each product to produce.

For example:

- T-shirts
- Pantalons
- Hoodies

Each product has different values:

- profit per unit
- pollution per unit
- production time
- resource usage
- demand limit

The objective is to find the best production plan while respecting the available production time, available resources, and product demand.

This project solves that problem using a metaheuristic optimization algorithm called **Grey Wolf Optimizer**.

---

## How the Optimizer Works

The application uses the **Grey Wolf Optimizer**, inspired by the hunting behavior and hierarchy of grey wolves.

The algorithm ranks solutions as:

| Role | Meaning |
|---|---|
| Alpha | Best solution |
| Beta | Second-best solution |
| Delta | Third-best solution |
| Omega | Remaining solutions |

Each wolf represents a possible production plan.

Example:

```text
[20, 15, 8]

<p align="center"><img src="https://github.com/user-attachments/assets/4edb7413-6107-47b7-a814-cfc33d30bd32" alt="Interface Screenshot" width="900"/></p>
