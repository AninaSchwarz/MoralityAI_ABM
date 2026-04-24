# MoralityAI_ABM

A simplified agent-based model of moral trade-offs in AI governance using a networked Hegselmann–Krause bounded-confidence model with adaptive confidence thresholds.

## Overview

This project simulates how surveillance, inference, safeguards, perceived threat, and contextual norm violations can affect opinion dynamics and polarization in a population.

The model is designed as a conceptual agent-based model. It does not use empirical data. Instead, it illustrates possible mechanisms through which privacy-related norm violations may reduce openness to disagreement and increase polarization, while surveillance and inference may reduce incident rates.

## Research Question

How can AI governance choices create trade-offs between security benefits and privacy-related moral harms?

The model focuses on the following question:

> Under what conditions can surveillance and inference reduce incidents while also increasing contextual privacy violations and polarization?

## Model Description

The script implements a simplified networked Hegselmann–Krause model.

Agents are placed in an undirected Erdős–Rényi social network. Each agent has an opinion value and a confidence bound, epsilon. At each time step, agents average the opinions of neighboring agents whose opinions fall within their confidence bound.

The confidence bound is adaptive:

- Higher perceived safety can increase openness.
- Contextual norm violations shrink openness.
- Lower openness limits interaction across opinion differences.
- Reduced interaction can increase or maintain polarization.

The model includes five main mechanisms:

1. Incident generation and perceived threat
2. Norm/context violation
3. Adaptive confidence bound `epsilon`
4. Networked HK opinion averaging
5. Polarization measurement

## Key Parameters

### `S` — Surveillance level

Represents the intensity of monitoring or data collection.

### `I` — Inference level

Represents the intensity of AI-based inference or prediction.

### `G` — Safeguards

Represents governance safeguards.

In the simplified model, safeguards reduce norm/context violations, but also slightly reduce the effectiveness of surveillance and inference in lowering incidents.

