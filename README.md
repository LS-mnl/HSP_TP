# Hardware for Signal Processing
Implémentation d'un CNN - LeNet-5 sur GPU

Réalisé par : **Manel KEBIR  & Rayane KHERZI**

## Getting Started

- Le langage utilisé pour ce TP est du Cuda, il vous faudra donc une carte graphique Nvidia. Sinon,ce n'est pas possible d'utiliser les fonctions qui utilisent le GPU. 
- Cuda n'est encore présent sur aucun IDE, mais comme la compilation et l'exécution se font via la console, il est possible d'utiliser n'importe quel IDE dans notre cas on a utilisé VsCode. 

#### Compilation et Execution depuis la console

Pour compiler un code Cuda, il vous suffit de lancer la commande : 

```
nvcc nomdufichier.cu -o nomdufichier
```

puis on a l'apparition d'un fichier portant le nom "nomdufichier". 
On exécute ce dernier, en Lancant la commande : 

```
./nomdufichier
```


<img src="https://img.shields.io/badge/1-Objectifs-red" alt="Objectifs" style="height: 40px;"/>

Les objectifs de ce projet sont : 
* Apprendre à utiliser CUDA
* Etudier la complexité d'algorithmes et l'accélération obtenue sur GPU par rapport à une exécution sur CPU
* Observer les limites de l'utilisation d'un GPU
* Implémenter "from scratch" un CNN : juste la partie inférence et non l'entrainement
* Exporter des données depuis un notebook python et les réimporter dans un projet cuda
* Faire un suivi de votre projet et du versionning grâce à l'outil git

### Implémentation d'un CNN
#### LeNet-5
A terme, l'objectif final est d'implémenter l'inférence d'un CNN très classique : LeNet-5.

La lecture de cet article apporte les informations nécessaires pour comprendre ce réseau de neurones.

https://www.datasciencecentral.com/profiles/blogs/lenet-5-a-classic-cnn-architecture

![LeNet-5](https://user-images.githubusercontent.com/75682374/149537925-7facd228-d0c7-4f33-b43d-17e5f4afd91f.png)

<img src="https://img.shields.io/badge/2-Prise en main de Cuda : Addition et Multiplication de matrices-red" alt="Prise en main de Cuda : Addition et Multiplication de matrices" style="height: 40px;"/>

### 2.1. Création de matrice
Dans tout le projet, que ce soit sur CPU ou GPU, on souhaite représenter les matrices sous forme de listes constituées des lignes de la matrice.
Dans la fonction d'initialisation, on initialise la matrice **NxP** avec des valeurs aléatoires.
![image](https://user-images.githubusercontent.com/94063629/148687288-f1d8b3a1-a6b9-4ab7-af0a-11b4140eb562.png)

### 2.2. Affichage de matrice sous forme conventionnelle
La création d'une fonction d’affichage de matrice sous sa forme classique, avec chaque ligne affichée l’une en dessous l’autre, est utile, voire même nécessaire afin de pouvoir vérifier le bon fonctionnement des opérations traitées ou prendre connaissance d'un résultat.
![image](https://user-images.githubusercontent.com/94063629/148687259-6fef5698-6d92-4c96-a2bf-ea60d089cf7e.png)

### 2.3. Addition
#### **CPU**
Sur **CPU**, on additionne deux matrices simplement comme à notre habitude en sommant les coefficients de chacune deux à deux puisque la représentation sous forme de liste n'est pas un frein à cette addition classique.
![image](https://user-images.githubusercontent.com/94063629/148687251-c18e9d34-435b-428f-af89-cd3c93302b13.png)

Ci-dessous un exemple de réalisation sur une addition de matrices 3x3:
![image](https://user-images.githubusercontent.com/94063629/149382058-443e7565-0896-4b1a-8a06-c49b4185ed1f.png)

#### **GPU**
Sur **GPU**, le calcul se base également sur la somme des coefficients deux à deux mais est un peu plus complexe à mettre en oeuvre en raison de la parallélisation des calculs sur GPU.
En particulier, la définition des indices des coefficients matriciels se fait cette fois via les variables _dim3_ définissant les threads afin de paralléliser (et donc accélérer) les calculs).
![image](https://user-images.githubusercontent.com/94063629/148687472-8454bbe7-95fc-4721-b2ce-a9f924f99767.png)

On définira par exemple l'indice de la ligne à considérer (au début de la fonction, et qui changera à chaque itération) par:
```
int lig = blockIdx.y * blockDim.y + threadIdx.y
```
où **_blockIdx.y_** désigne le numéro de la ligne du _block_ dans la _grid_,

   **_blockDim.y_** donne le nombre de lignes total de ce _block_,
   
   **_threadIdx.y_** donne le numéro de la ligne du _thread_ appartenant au _block_.

**Ce qui suit est nécessaire pour tout calcul effectué sur le GPU, et est donc également nécessaire pour la réalisation de la multiplication.**

Il est, en outre, nécessaire de définir la fonction avec __global__ afin d’effectuer les calculs sur le GPU en l’appelant depuis le CPU, par exemple dans le cas de l'addition de matrices sur GPU:
```
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p);
```
Ce sera le cas pour toutes les fonctions effectuant des calculs sur le GPU tout en étant appelé depuis le CPU.

Les mémoires des matrices doivent être allouées sur le GPU avec ```cudaMalloc``` et copiées depuis le CPU vers le GPU avec ```cudaMemcpy```
Pour que la parallélisation soit fonctionnelle et efficace il est nécessaire de définir **en dehors de la fonction** (dans le main par exemple) les dimensions des variables _dim3_.
```
dim3 block_size(n, m);
dim3 grid_size(1, 1);
```
où _**n**_ et _**m**_ sont les dimensions de la matrice résultante du calcul.

Enfin, la fonction d'addition sur le GPU est appelée via la commande suivante:
```
cudaMatrixAdd<<<grid_size, block_size>>>(d_M1, d_M2, d_Mout, n, p);
```
Ci-dessous un exemple de réalisation:

![image](https://user-images.githubusercontent.com/94063629/149382120-fc89343c-9070-483d-a279-e6ec71d3646f.png)

### 2.4. Multiplication
#### **CPU**
La multiplication de deux matrices sur le CPU se fait de façon habituelle. La seule difficulté réside dans l'indexage correct des coefficients recherchés, les matrices étant sous forme de liste.
![image](https://user-images.githubusercontent.com/94063629/148688381-221ddec3-26b4-46ba-b3d4-df48913f3031.png)

Ci-dessous un exemple de réalisation sur une multiplication de matrices 3x3:
![image](https://user-images.githubusercontent.com/94063629/149382363-1447e7d3-da39-42f4-8886-212a06a61db2.png)

#### **GPU**
Comme pour l'addition, la multiplication sur GPU repose sur le même principe que la multiplication classique mais les indexes lignes et colonnes désirées doivent être définies (exactement selon la même formulation que pour l'addition) avec les variables définissant les threads et blocks.

Ci-dessous un exemple de réalisation sur une multiplication de matrices 3x3:
![image](https://user-images.githubusercontent.com/94063629/149382451-8b1d5a11-f984-4b3e-8e46-3f14cc048176.png)


<img src="https://img.shields.io/badge/3-Partie 2. Premières couches du réseau de neurones LeNet-5 : Convolution 2D et subsampling-red" alt="Partie 2. Premières couches du réseau de neurones LeNet-5 : Convolution 2D et subsampling" style="height: 50px;"/>
## 3- Partie 2. Premières couches du réseau de neurones LeNet-5 : Convolution 2D et subsampling
L'architecture du réseau LeNet-5 est composée de plusieurs couches :

* Layer 1- Couche d'entrée de taille 32x32 correspondant à la taille des images de la base de données MNIST
* Layer 2- Convolution avec 6 noyaux de convolution de taille 5x5. La taille résultante est donc de 6x28x28.
* Layer 3- Sous-échantillonnage d'un facteur 2. La taille résultante des données est donc de 6x14x14.

### 3.1. Layer 1 - Génération des données de tests
La génération des données consiste en la création des matrices suivantes sous la forme de tableaux à une dimension:
- la matrice d'entrée dans le réseau 32x32 **raw_data** initialisée avec des valeurs aléatoires entre 0 et 1.
- la matrice 6x28x28 **C1_data** résultante de la convolution 2D initialisée avec des valeurs nulles.
- la matrice 6x14x14 **S1_data** résultante du sous-échantillonnage initialisée avec des valeurs nulles.
- le kernel 6x5x5 **C1_kernel** permettant la convolution de la layer 1 et initialisé entre 0 et 1.

### 3.2. Layer 2 - Convolution 2D
La convolution se fait exclusivement sur le GPU. De façon analogue à la multiplication, on fait glisser un kernel **C1_kernel** sur la totalité de la matrice **raw_data** pour obtenir la matrice résultante **C1_data**.

![Peek 2022-01-10 00-30](https://user-images.githubusercontent.com/75682374/148705636-8c98babd-8159-4ac1-b08b-8abeef920215.gif)

Il est également nécessaire de prendre un compte le nombre de kernel (la profondeur de **C1_kernel**) sur les calculs de convolution afin d'obtenir le nombre de _features maps_ souhaités.

### 3.3. Layer 3 - Sous-échantillonnage
Le sous-échantillonage se fait par une fonction de _MeanPooling_, à savoir un moyennage sur une fenêtre glissante 2x2 (afin de réduire par 2 les dimensions de **raw_data** et d'obtenir **S1_data**).
![image](https://user-images.githubusercontent.com/94063629/148841381-37243479-3f74-4a92-afff-a094cc323501.png)
Celui-ci se fait également sur le GPU depuis un appel du CPU.

### 3.4. Fonctions d'activation
Dans l'objectif de parfaire le réseau de neurones, une couche d'activation est requise. Comme on peut le remarquer dans l'article sur l'architecture de LeNet-5, la fonction d'activation utilisée est une tangente hyperbolique (_tanh_). Celle-ci interviendra après chaque layer de _Conv2D_.
Afin de se laisser la possibilité d'appeler cette fonction d'activation depuis chaque kernel sur le GPU, on définit cette fois la fonction avec le _specifier_ ```__device__```, et non ```__global__``` pour effectuer les calculs sur le GPU depuis un appel du GPU.
La couche de fonction d'activation retourne une matrice de même dimension que celle qui lui est fournie.

### 3.5 Exemple
Voci ci-dessous un exemple de réalisation d'une convolution d'une matrice 8x8 par un kernel 5x5 suivie éventuellement d'une fonction d'activation, puis d'un _MeanPooling_:
![image](https://user-images.githubusercontent.com/94063629/149505445-78691e54-997d-4050-af17-7a51034525a7.png)
Dans cet exemple, on choisit volontairement des matrices aux coefficients simples afin de pouvoir confirmer le bon déroulement des calculs: le kernel nul avec un 2 central permet notamment la vérification rapide des calculs sur une matrice unitaire.

## 4- Partie 3. Un peu de Python

### 4.1. Notebook Python
Dans cette dernière partie, on utilise le notebook Python comme référence afin de finaliser notre réseau LeNet5.
En particulier, celui-ci nous servira, grâce à un entraînement rapide, d'obtenir les valeurs optimales des poids de chaque couche afin de pouvoir initialiser les _kernels_ de convolution et les poids des couches _fully connected_ de façon à obtenir les meilleurs résultats.

En effet, dans ce projet, on ne désire pas créer la fonction d'entrainement du réseau de neurones en Cuda, car cela est beaucoup plus complexe, et nous aurait pris trop de temps à mettre en place (*Descente de gradient, BackPropagation...*). Ceci pourrait donc être un bon point de départ pour continuer ce projet :
* Créer l'optimizer
* La fonction de loss
* L'entrainement sur une base de test et de validation.

### 4.2. Création des fonctions manquantes
On construit le réseau en ajoutant couches de convolution et de _MeanPooling_. Il est également nécessaire de créer une couche de _Dense_ effectuant l'opération **W.x + b** où W sont les poids et b, les biais appliqués à l'image d'entrée x.
Cette fonction fait intervenir les fonctions de multiplication et d'addition sur le GPU.

En outre, afin de prévoir les cas où les matrices **W** et **x** ne sont pas carrées, on se propose d'introduire une nouvelle fonction de multiplication (nommée _cudaMatrixMultGeneral_), basée sur le même principe que celle créée plus haut sur GPU, mais effectuant la multiplication d'une matrice **NxP** par une matrice **PxM** pour donner une matrice résultante **NxM** (celle-ci se trouve dans le fichier _Partie3.cu_).

### 4.3. Export des poids dans un fichier .h5
Après avoir entraîné le réseau dans le notebook et récupéré les poids et biais de chaque couche, on les utilise pour initialiser les _kernels_. 

Le réseau LeNet5 est désormais entièrement fonctionnel pour une image d'entrée de dimensions 32x32.


Cette partie n'est pas encore complètement terminée. La création du réseau est faite et fonctionne au vu des différents tests, mais la liaison avec les poids pas encore... Il reste à finir cette partie à finir pour que le réseau puisse être totalement fonctionnel. 


