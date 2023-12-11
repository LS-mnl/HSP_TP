# Hardware for Signal Processing
Implémentation d'un CNN - LeNet-5 sur GPU

Réalisé par : **KEBIR Manel KHERZI Rayane**

## Getting Started

### Prérequis

Si vous voulez continuer ce projet, ou bien utiliser une partie de celui-ci, vous n'aurez pas besoin de grand chose. 
Le langage utilisé pour ce projet est du Cuda, il vous faudra donc a minima un PC avec une carte graphique Nvidia. Sinon, vous ne pourrez pas utiliser les fonctions qui utilisent le GPU. 

#### Quel IDE pour le langage Cuda ? 
Aujourd'hui le langage Cuda n'est encore présent sur aucun IDE, mais comme la compilation et l'exécution se font via la console, il est possible d'utiliser n'importe quel IDE. 

Un IDE comprenant la coloration synthaxique du C ou du C++ fait largement l'affaire. Choisissez donc celui qui vous voulez (Jupyter-Lab, VsCode ou encore Sublime Text font largement l'affaire)


#### Compilation et Execution depuis la console

Pour compiler un code Cuda, il vous suffit de lancer la commande : 

```
nvcc nomdufichier.cu -o nomdufichier
```

Quand vous aurez fait cela, vous verrez apparaître un fichier portant le nom "nomdufichier". 
Vous n'avez donc plus qu'à l'exécuter, et là encore, rien de plus simple. Lancer simplement la commande : 

```
./nomdufichier
```

PS: Pour que ces commandes fonctionnent il faut bien sûr que vous soyez dans votre dossier de travail. Vous pouvez vous déplacer facilement dans les dossiers grâce à la commande ```cd```.

