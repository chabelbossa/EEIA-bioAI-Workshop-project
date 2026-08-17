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
    """
    # 1. Perte sur les étiquettes dures (0/1)
    hard_loss = F.binary_cross_entropy_with_logits(student_logits, labels.float())

    # 2. Calcul des prédictions adoucies (soft targets)
    # torch.sigmoid convertit les logits 1D en probabilités (classification binaire)
    # La température divise les logits avant la sigmoïde, ce qui "adoucit" la distribution
    t_soft = torch.sigmoid(teacher_logits / temperature)
    s_soft = torch.sigmoid(student_logits / temperature)

    # 3. Perte de connaissance distillée (Cross-Entropy sur soft targets)
    # On utilise binary_cross_entropy car nos prédictions sont des probabilités entre 0 et 1
    soft_loss = F.binary_cross_entropy(s_soft, t_soft) * (temperature ** 2)

    # 4. Perte totale combinée
    # alpha pondère l'importance du loss supervisé (hard_loss) vs le loss de distillation (soft_loss)
    loss = alpha * hard_loss + (1 - alpha) * soft_loss
    
    return loss
 

def train_student(student, teacher_logits, inputs, labels, epochs=20, lr=1e-3,
                   temperature=4.0, alpha=0.5, device="cpu", batch_size=256,
                   log_every=10):
    """inputs: tenseur déjà transformé en caractéristiques pour le student (vecteurs
    k-mer ou fenêtres one-hot). teacher_logits: précalculés une fois, dans le même ordre
    que inputs.
    """
    # 1. Configuration de l'entraînement
    student.to(device)  # Déplace le modèle sur le GPU/CPU
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    n = len(inputs)
    history = {"loss": [], "accuracy": []}
    
    # Boucle principale d'entraînement
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        indices = torch.randperm(n).tolist()  # Mélange des indices pour chaque époque
        
        # On itère sur les données par mini-lots
        for i in range(0, n, batch_size):
            # Récupération du batch
            idx = indices[i : i + batch_size]
            
            # 2. Préparation des données du batch (extraction et déplacement vers device)
            x = inputs[idx].to(device)
            y = labels[idx].to(device)
            t_logits = teacher_logits[idx].to(device)
            
            # 3. Rétropropagation
            optimizer.zero_grad()
            s_logits = student(x)
            loss = distillation_loss(s_logits, t_logits, y, temperature, alpha)
            loss.backward()
            optimizer.step()
            
            # Accumulation des métriques
            epoch_loss += loss.item() * len(idx)
            # On prédit en prenant le seuil de 0 (s_logits > 0 équivaut à p > 0.5)
            epoch_correct += ((s_logits > 0).float() == y).sum().item()
            
        # 4. Calcul des moyennes et enregistrement dans l'historique
        avg_loss = epoch_loss / n
        acc = epoch_correct / n
        
        history["loss"].append(avg_loss)
        history["accuracy"].append(acc)
        
        # 5. Affichage du log (toutes les 'log_every' époques)
        if (epoch + 1) % log_every == 0 or epoch == 0 or epoch == epochs - 1:
            print(f"epoch {epoch + 1:3d}/{epochs} | loss={avg_loss:.4f} | acc={acc:.4f}")
            
    return student, history
