# EEIA — bioAI Workshop
# Author: Généreux Akotenou — PhD student, BioinformaticLabs College of Computing / UM6P
# https://github.com/Genereux-akotenou/EEIA-bioAI-Workshop
# Licensed under the MIT License. See LICENSE at the repo root.

"""
distillation.py — teacher (embedding Evo2 gelé + MLPHead) -> student minuscule (MLP sur
k-mers), via cibles douces + température (perte hybride).

**À COMPLÉTER** : `distillation_loss` et `train_student` ci-dessous.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def distillation_loss(student_logits, teacher_logits, labels,
                       temperature: float = 4.0, alpha: float = 0.5):
    """Perte hybride = alpha * CE sur étiquettes dures + (1 - alpha) * KD sur cibles douces.

    alpha=1.0 est un entraînement supervisé classique (pas de distillation) ;
    alpha=0.0 est une pure imitation des cibles douces du teacher.

    TODO :
      1. `hard_loss` : binary_cross_entropy_with_logits(student_logits, labels.float())
      2. `t_soft` : sigmoid(teacher_logits / temperature) — la prédiction "adoucie" du teacher
      3. `s_soft` : sigmoid(student_logits / temperature) — la prédiction adoucie du student
      4. `soft_loss` : binary_cross_entropy(s_soft, t_soft) * (temperature ** 2)
         (le facteur temperature**2 compense l'échelle des gradients rétrécis par la
         division par la température)
      5. retournez alpha * hard_loss + (1 - alpha) * soft_loss
    """
    raise NotImplementedError("TODO : implémentez distillation_loss")


def train_student(student, teacher_logits, inputs, labels, epochs=20, lr=1e-3,
                   temperature=4.0, alpha=0.5, device="cpu", batch_size=256):
    """inputs: tenseur déjà transformé en caractéristiques pour le student (vecteurs
    k-mer ou fenêtres one-hot). teacher_logits: précalculés une fois, dans le même ordre
    que inputs.

    TODO :
      1. Placez `student` sur `device`, créez un optimiseur Adam (lr=lr).
      2. Pour chaque époque, mélangez les indices (torch.randperm), puis pour chaque
         mini-lot de taille batch_size :
           - récupérez x, y, t_logits pour ce mini-lot (et déplacez-les sur `device`)
           - remettez les gradients à zéro
           - calculez s_logits = student(x)
           - calculez la perte avec distillation_loss(s_logits, t_logits, y, temperature, alpha)
           - rétropropagez, faites un pas d'optimisation
           - accumulez epoch_loss += loss.item() * len(idx)
      3. Ajoutez la perte moyenne de l'époque à history["loss"]
      4. Retournez (student, history)
    """
    history = {"loss": []}
    raise NotImplementedError("TODO : implémentez train_student")
