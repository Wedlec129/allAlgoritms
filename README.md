# allAlgoritms


все основные алгоримы которые должен знать програмист.





[Алгоритмы сортировки ]

-[Пузырьковая сортировка ](#1)

-[Сортировка выбора ](#2)

-[Сортировка вставками ](#3)

-[Сортировка слиянием ](#4)

-[Быстрая сортировка ](#5)

-[Сортировка по основанию ](#6)


[Binary Search ](#7)

[Building Trees ](#8)


[Линейный поиск ]

-[Линейный поиск ](#9)

-[Поиск в глубину (DFS) ](#10)

-[Поиск в ширину (BFS) ](#11)

-[Бинарный поиск ](#12)

-[Двоичное дерево поиска (BST) ](#13)


[Динамическое программирование ]

-[Динамические массивы ](#14)

-[Хранение нескольких классов ](#18)

-[Правильное хранение нескольких классов ](#19)

-[Передача динамических объектов в функции ](#20)

-[Создание универсальных функций ](#21)




[Алгоритмы обхода графа]

-[Алгоритм Дейкстры ](#22)

-[Алгоритм Беллмана-Форда ](#23)

-[Алгоритм Крускала ](#24)

-[Алгоритм Прима ](#25)








## <a id="1">**a)Пузырьковая сортировка:**</a> 

- **Описание:** Сравнивает соседние элементы и меняет их местами, если они расположены в неправильном порядке.
- **Скорость выполнения:** средняя и наихудшая временная сложность O(n^2). Простой, но неэффективный.
- **Код С++:**

```cpp
#include <iostream>

void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                std::swap(arr[j], arr[j + 1]);
            }
        }
    }
}

int main() {
    int arr[] = {5, 2, 8, 12, 1};
    int n = sizeof(arr) / sizeof(arr[0]);

    std::cout << "Original array: ";
    for (int i = 0; i < n; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    bubbleSort(arr, n);

    return 0;
}

```



**Код Swift:**

```swift
func bubbleSort(_ arr: inout [Int]) {
    let n = arr.count
    for i in 0..<n {
        for j in 0..<n-i-1 {
            if arr[j] > arr[j+1] {
                arr.swapAt(j, j+1)
            }
        }
    }
}

var arr = [5, 2, 8, 12, 1]
bubbleSort(&arr)
print("Sorted array: \(arr)")

```



**Код Python:**

```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

arr = [5, 2, 8, 12, 1]
bubble_sort(arr)
print("Sorted array:", arr)

```



## <a id="2"></a>

**b) Сортировка выбора:**

- **Описание:** Выбирает минимальный элемент из несортированной части массива и меняет его местами с первым элементом.
- **Скорость выполнения:** средняя и наихудшая временная сложность O(n^2). Простой, но неэффективный.
- **Код С++:**

```cpp
#include <iostream>

void selectionSort(int arr[], int n) {
    for (int i = 0; i < n - 1; ++i) {
        int minIndex = i;
        for (int j = i + 1; j < n; ++j) {
            if (arr[j] < arr[minIndex]) {
                minIndex = j;
            }
        }
        std::swap(arr[i], arr[minIndex]);
    }
}



int main() {
    int arr[] = {5, 2, 8, 12, 1};
    int n = sizeof(arr) / sizeof(arr[0]);

    std::cout << "Original array: ";
    for (int i = 0; i < n; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    selectionSort(arr, n);

    return 0;
}


```



**Код Swift:**

```swift
func selectionSort(_ arr: inout [Int]) {
    let n = arr.count
    for i in 0..<n {
        var minIndex = i
        for j in i+1..<n {
            if arr[j] < arr[minIndex] {
                minIndex = j
            }
        }
        arr.swapAt(i, minIndex)
    }
}

var arr = [5, 2, 8, 12, 1]
selectionSort(&arr)
print("Sorted array: \(arr)")

```



**Код Python:**

```python
def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_index = i
        for j in range(i+1, n):
            if arr[j] < arr[min_index]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]

arr = [5, 2, 8, 12, 1]
selection_sort(arr)
print("Sorted array:", arr)

```




## <a id="3"></a>
**Сортировка вставками:**

- **Описание:** Создает окончательный отсортированный массив по одному элементу за раз, вставляя каждый элемент в правильное положение.
- **Скорость выполнения:** средняя и наихудшая временная сложность O(n^2). Эффективен для небольших массивов или частично отсортированных массивов.
- **Код С++:**

```cpp

#include <iostream>
void insertionSort(int arr[], int n) {
    for (int i = 1; i < n; ++i) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            --j;
        }
        arr[j + 1] = key;
    }
}



int main() {
    int arr[] = {5, 2, 8, 12, 1};
    int n = sizeof(arr) / sizeof(arr[0]);

    std::cout << "Original array: ";
    for (int i = 0; i < n; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    insertionSort(arr, n);

    return 0;
}
```



**Код Swift:**

```swift
func insertionSort(_ arr: inout [Int]) {
    let n = arr.count
    for i in 1..<n {
        let key = arr[i]
        var j = i-1
        while j >= 0 && arr[j] > key {
            arr[j+1] = arr[j]
            j -= 1
        }
        arr[j+1] = key
    }
}

var arr = [5, 2, 8, 12, 1]
insertionSort(&arr)
print("Sorted array: \(arr)")

```



**Код Python:**

```python
def insertion_sort(arr):
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key

arr = [5, 2, 8, 12, 1]
insertion_sort(arr)
print("Sorted array:", arr)

```




## <a id="4"></a>
**d) Сортировка слиянием:**

- **Описание:** Делит массив на две половины, рекурсивно сортирует их, а затем объединяет две отсортированные половины.
- **Скорость выполнения:** средняя, лучшая и наихудшая временная сложность O(n log n). Эффективный и стабильный.
- **Код С++:**

```cpp
#include <iostream>
void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;
    int L[n1], R[n2];

    for (int i = 0; i < n1; ++i) {
        L[i] = arr[left + i];
    }
    for (int j = 0; j < n2; ++j) {
        R[j] = arr[mid + 1 + j];
    }

    int i = 0;
    int j = 0;
    int k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            ++i;
        } else {
            arr[k] = R[j];
            ++j;
        }
        ++k;
    }

    while (i < n1) {
        arr[k] = L[i];
        ++i;
        ++k;
    }

    while (j < n2) {


        arr[k] = R[j];
        ++j;
        ++k;
    }
}

void mergeSort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}


int main() {
    int arr[] = {5, 2, 8, 12, 1};
    int n = sizeof(arr) / sizeof(arr[0]);

    std::cout << "Original array: ";
    for (int i = 0; i < n; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    mergeSort(arr, 0, n - 1);

    return 0;
}
```



**Код Swift:**

```swift
func mergeSort(_ arr: [Int]) -> [Int] {
    guard arr.count > 1 else { return arr }
    
    let mid = arr.count / 2
    let leftArray = mergeSort(Array(arr[..<mid]))
    let rightArray = mergeSort(Array(arr[mid...]))
    
    return merge(leftArray, rightArray)
}

func merge(_ left: [Int], _ right: [Int]) -> [Int] {
    var merged = [Int]()
    var leftIndex = 0
    var rightIndex = 0
    
    while leftIndex < left.count && rightIndex < right.count {
        if left[leftIndex] < right[rightIndex] {
            merged.append(left[leftIndex])
            leftIndex += 1
        } else {
            merged.append(right[rightIndex])
            rightIndex += 1
        }
    }
    
    return merged + Array(left[leftIndex...]) + Array(right[rightIndex...])
}

let arr = [5, 2, 8, 12, 1]
let sortedArr = mergeSort(arr)
print("Sorted array: \(sortedArr)")

```



**Код Python:**

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)

def merge(left, right):
    merged = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    
    merged.extend(left[i:])
    merged.extend(right[j:])
    
    return merged

arr = [5, 2, 8, 12, 1]
sorted_arr = merge_sort(arr)
print("Sorted array:", sorted_arr)

```



## <a id="5"></a>

**e) Быстрая сортировка:**

- **Описание:** Делит массив на более мелкие подмассивы на основе элемента сводки, а затем рекурсивно сортирует подмассивы.
- **Скорость выполнения:** Временная сложность в среднем случае O(n log n), временная сложность в наихудшем случае O(n^2). Эффективен на практике.
- **Код С++:**

```cpp
#include <iostream>
int partition(int arr[], int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; ++j) {
        if (arr[j] < pivot) {
            ++i;
            std::swap(arr[i], arr[j]);
        }
    }
    std::swap(arr[i + 1], arr[high]);
    return i + 1;
}

void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

int main() {
    int arr[] = {5, 2, 8, 12, 1};
    int n = sizeof(arr) / sizeof(arr[0]);

    std::cout << "Original array: ";
    for (int i = 0; i < n; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    quickSort(arr, 0, n - 1);

    return 0;
}
```



**Код Swift:**

```swift
func quickSort(_ arr: inout [Int], low: Int, high: Int) {
    guard low < high else { return }
    
    let partitionIndex = partition(&arr, low: low, high: high)
    
    quickSort(&arr, low: low, high: partitionIndex - 1)
    quickSort(&arr, low: partitionIndex + 1, high: high)
}

func partition(_ arr: inout [Int], low: Int, high: Int) -> Int {
    let pivot = arr[high]
    var i = low
    
    for j in low..<high {
        if arr[j] <= pivot {
            arr.swapAt(i, j)
            i += 1
        }
    }
    
    arr.swapAt(i, high)
    return i
}

var arr = [5, 2, 8, 12, 1]
quickSort(&arr, low: 0, high: arr.count - 1)
print("Sorted array: \(arr)")

```



**Код Python:**

```python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    smaller, equal, larger = [], [], []
    
    for num in arr:
        if num < pivot:
            smaller.append(num)
        elif num == pivot:
            equal.append(num)
        else:
            larger.append(num)
    
    return quick_sort(smaller) + equal + quick_sort(larger)

arr = [5, 2, 8, 12, 1]
sorted_arr = quick_sort(arr)
print("Sorted array:", sorted_arr)

```




## <a id="6"></a>
**f) Сортировка по основанию:**

- **Описание:** Сортирует целые числа, группируя числа по отдельным цифрам, от наименее значащих до наиболее значимых.
- **Скорость выполнения:** Временная сложность O(d * (n + b)), где d — количество цифр, n — количество элементов, а b — основание (обычно 10).
- **Код С++:**

```cpp
#include <iostream>
int getMax(int arr[], int n) {
    int max = arr[0];
    for (int i = 1; i < n; ++i) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

void countSort(int arr[], int n, int exp) {
    int output[n];
    int count[10] = {0};

    for (int i = 0; i < n; ++i) {
        ++count[(arr[i] / exp) % 10];
    }

    for (int i = 1; i < 10; ++i) {
        count[i] += count[i - 1];
    }

    for (int i = n - 1; i >= 0; --i) {
        output[count[(arr[i] / exp) % 10] - 1] = arr[i];
        --count[(arr[i] / exp) % 10];
    }

    for (int i = 0; i < n; ++i) {
        arr[i] = output[i];
    }
}

void radixSort(int arr[], int n) {
    int max = getMax(arr, n);
    for (int exp = 1; max / exp > 0; exp *= 10) {
        countSort(arr, n, exp);
    }
}

int main() {
    int arr[] = {5, 2, 8, 12, 1};
    int n = sizeof(arr) / sizeof(arr[0]);

    std::cout << "Original array: ";
    for (int i = 0; i < n; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << std::endl;

    radixSort(arr, n);

    return 0;
}


```



**Код Swift:**

```swift
func radixSort(_ arr: inout [Int]) {
    let maxElement = arr.max() ?? 0
    var divisor = 1
    
    while divisor <= maxElement {
        var buckets: [[Int]] = Array(repeating: [], count: 10)
        
        for num in arr {
            let digit = (num / divisor) % 10
            buckets[digit].append(num)
        }
        
        arr = buckets.flatMap { $0 }
        divisor *= 10
    }
}

var arr = [5, 2, 8, 12, 1]
radixSort(&arr)
print("Sorted array: \(arr)")

```



**Код Python:**

```python
def radix_sort(arr):
    max_element = max(arr)
    exp = 1
    while max_element // exp > 0:
        counting_sort(arr, exp)
        exp *= 10

def counting_sort(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    
    for num in arr:
        index = num // exp % 10
        count[index] += 1
    
    for i in range(1, 10):
        count[i] += count[i-1]
    
    i = n - 1
    while i >= 0:
        index = arr[i] // exp % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1
        i -= 1
    
    for i in range(n):
        arr[i] = output[i]

arr = [5, 2, 8, 12, 1]
radix_sort(arr)
print("Sorted array:", arr)

```




## <a id="7"></a>
### Binary Search:

- **Описание:** Поиск целевого элемента в отсортированном массиве путем многократного деления интервала поиска пополам.
- **Скорость выполнения:** Средняя и наихудшая временная сложность O(log n).
- **Код С++:**

```cpp
#include <iostream>
int binarySearch(int arr[], int target, int low, int high) {
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return -1;
}

int main() {
    int arr[] = {1, 5, 8, 12, 15};
    int n = sizeof(arr) / sizeof(arr[0]);
    int target = 8;

    int index = binarySearch(arr, target, 0, n - 1);
    if (index != -1) {
        std::cout << "Element found at index " << index << std::endl;
    } else {
        std::cout << "Element not found" << std::endl;
    }

    return 0;
}
```



**Код Swift:**

```swift
func binarySearch(_ arr: [Int], target: Int) -> Int? {
    var left = 0
    var right = arr.count - 1
    
    while left <= right {
        let mid = left + (right - left) / 2
        
        if arr[mid] == target {
            return mid
        }
        
        if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return nil
}

let arr = [1, 5, 8, 12, 15]
let target = 8

if let index = binarySearch(arr, target: target) {
    print("Element found at index \(index)")
} else {
    print("Element not found")
}

```



**Код Python:**

```python
def binary_search(arr, target):
    left = 0
    right = len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2
        
        if arr[mid] == target:
            return mid
        
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

arr = [1, 5, 8, 12, 15]
target = 8
result = binary_search(arr, target)
if result != -1:
    print("Element found at index", result)
else:
    print("Element not found")

```


## <a id="8"></a>
### Building Trees:

- #### Двоичное дерево поиска (BST):

  - **Описание:** бинарное дерево, в котором левый дочерний элемент меньше родительского, а правый дочерний элемент больше.
  - **Код С++:**
```cpp
#include <iostream>
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;

    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};

TreeNode* insert(TreeNode* node, int val) {
    if (!node) {
        return new TreeNode(val);
    }
    if (val < node->val) {
        node->left = insert(node->left, val);
    } else {
        node->right = insert(node->right, val);
    }
    return node;
}

int main() {
    TreeNode* root = nullptr;

    root = insert(root, 5);
    root = insert(root, 2);
    root = insert(root, 8);
    root = insert(root, 12);
    root = insert(root, 1);

    // Perform tree operations...

    return 0;
}
```



**Код Swift:**

```swift
class Node {
    var value: Int
    var left: Node?
    var right: Node?
    
    init(value: Int) {
        self.value = value
        self.left = nil
        self.right = nil
    }
}

class BST {
    var root: Node?
    
    func insert(value: Int) {
        root = insertRecursive(root, value: value)
    }
    
    private func insertRecursive(_ node: Node?, value: Int) -> Node {
        guard let node = node else {
            return Node(value: value)
        }
        
        if value < node.value {
            node.left = insertRecursive(node.left, value: value)
        } else if value > node.value {
            node.right = insertRecursive(node.right, value: value)
        }
        
        return node
    }
    
    // Other methods for tree traversal, deletion, etc.
}

// Usage example
let bst = BST()
bst.insert(value: 5)
bst.insert(value: 2)
bst.insert(value: 8)
bst.insert(value: 12)
bst.insert(value: 1)

```



**Код Python:**

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, value):
        self.root = self._insert_recursive(self.root, value)

    def _insert_recursive(self, node, value):
        if node is None:
            return Node(value)

        if value < node.value:
            node.left = self._insert_recursive(node.left, value)
        elif value > node.value:
            node.right = self._insert_recursive(node.right, value)
        
        return node

# Usage example
bst = BST()
bst.insert(5)
bst.insert(2)
bst.insert(8)
bst.insert(12)
bst.insert(1)

```


## <a id="9"></a>
## Линейный поиск

Линейный поиск — это простой алгоритм поиска, который последовательно проверяет каждый элемент в списке до тех пор, пока не будет найдено совпадение или не будет достигнут конец списка.

```cpp
#include <iostream>
#include <vector>

int linearSearch(const std::vector<int>& arr, int target) {
    for (int i = 0; i < arr.size(); ++i) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;
}

int main() {
    std::vector<int> arr = {5, 2, 8, 12, 1};
    int target = 8;
    int result = linearSearch(arr, target);
    
    if (result != -1) {
        std::cout << "Element found at index " << result << std::endl;
    } else {
        std::cout << "Element not found" << std::endl;
    }
    
    return 0;
}
```



swift

```swift
func linearSearch(_ arr: [Int], _ target: Int) -> Int {
    for (index, element) in arr.enumerated() {
        if element == target {
            return index
        }
    }
    return -1
}

func main() {
    let arr = [5, 2, 8, 12, 1]
    let target = 8
    let result = linearSearch(arr, target)
    
    if result != -1 {
        print("Element found at index", result)
    } else {
        print("Element not found")
    }
}

main()
```



python:

```python
def linear_search(arr, target):
    for index, element in enumerate(arr):
        if element == target:
            return index
    return -1

def main():
    arr = [5, 2, 8, 12, 1]
    target = 8
    result = linear_search(arr, target)
    
    if result != -1:
        print("Element found at index", result)
    else:
        print("Element not found")

main()
```



**Скорость**: линейный поиск имеет среднюю и наихудшую временную сложность O(n), где n — количество элементов в списке.
## <a id="10"></a>
## Поиск в глубину (DFS)

Поиск в глубину — это алгоритм обхода графа, который максимально исследует каждую ветвь перед возвратом.

```cpp
#include <iostream>
#include <vector>
#include <stack>

void dfs(const std::vector<std::vector<int>>& graph, int start) {
    std::vector<bool> visited(graph.size(), false);
    std::stack<int> stack;
    stack.push(start);
    
    while (!stack.empty()) {
        int node = stack.top();
        stack.pop();
        
        if (!visited[node]) {
            std::cout << node << " ";
            visited[node] = true;
            
            for (int neighbor : graph[node]) {
                if (!visited[neighbor]) {
                    stack.push(neighbor);
                }
            }
        }
    }
}

int main() {
    std::vector<std::vector<int>> graph = {
        {1, 2},
        {0, 2, 3},
        {0, 1, 4},
        {1, 4},
        {2, 3}
    };
    
    int startNode = 0;
    std::cout << "DFS traversal: ";
    dfs(graph, startNode);
    std::cout << std::endl;
    
    return 0;
}
```

swift

```swift
func dfs(_ graph: [[Int]], _ start: Int) {
    var visited = Array(repeating: false, count: graph.count)
    var stack = [start]
    
    while !stack.isEmpty {
        let node = stack.removeLast()
        
        if !visited[node] {
            print(node, terminator: " ")
            visited[node] = true
            
            for neighbor in graph[node] {
                if !visited[neighbor] {
                    stack.append(neighbor)
                }
            }
        }
    }
}

func main() {
    let graph = [
        [1, 2],
        [0, 2, 3],
        [0, 1, 4],
        [1, 4],
        [2, 3]
    ]
    
    let startNode = 0
    print("DFS traversal:", terminator: " ")
    dfs(graph, startNode)
    print()
}

main()
```



python:

```python
def dfs(graph, start):
    visited = [False] * len(graph)
    stack = [start]

    while stack:
        node = stack.pop()

        if not visited[node]:
            print(node, end=" ")
            visited[node] = True

            for neighbor in graph[node]:
                if not visited[neighbor]:
                    stack.append(neighbor)

def main():
    graph = [
        [1, 2],
        [0, 2, 3],
        [0, 1, 4],
        [1, 4],
        [2, 3]
    ]

    start_node = 0
    print("DFS traversal:", end=" ")
    dfs(graph, start_node)
    print()

main()
```



**Скорость**: временная сложность DFS составляет O(V + E), где V — количество вершин, а E — количество ребер в графе.

<a id="11"></a>

## Поиск в ширину (BFS)

Поиск в ширину — это алгоритм обхода графа, который исследует все вершины графа в движении вширь, начиная с заданной исходной вершины.

```cpp
#include <iostream>
#include <vector>
#include <queue>

void bfs(const std::vector<std::vector<int>>& graph, int start) {
    std::vector<bool> visited(graph.size(), false);
    std::queue<int> queue;
    queue.push(start);
    visited[start] = true;
    
    while (!queue.empty()) {
        int node = queue.front();
        queue.pop();
        std::cout << node << " ";
        
        for (int neighbor : graph[node]) {
            if (!visited[neighbor]) {
                queue.push(neighbor);
                visited[neighbor] = true;
            }
        }
    }
}

int main() {
    std::vector<std::vector

<int>> graph = {
        {1, 2},
        {0, 2, 3},
        {0, 1, 4},
        {1, 4},
        {2, 3}
    };
    
    int startNode = 0;
    std::cout << "BFS traversal: ";
    bfs(graph, startNode);
    std::cout << std::endl;
    
    return 0;
}
```

swift

```swift
func bfs(_ graph: [[Int]], _ start: Int) {
    var visited = Array(repeating: false, count: graph.count)
    var queue = [start]
    visited[start] = true
    
    while !queue.isEmpty {
        let node = queue.removeFirst()
        print(node, terminator: " ")
        
        for neighbor in graph[node] {
            if !visited[neighbor] {
                queue.append(neighbor)
                visited[neighbor] = true
            }
        }
    }
}

func main() {
    let graph = [
        [1, 2],
        [0, 2, 3],
        [0, 1, 4],
        [1, 4],
        [2, 3]
    ]
    
    let startNode = 0
    print("BFS traversal:", terminator: " ")
    bfs(graph, startNode)
    print()
}

main()
```



python:

```python
from collections import deque

def bfs(graph, start):
    visited = [False] * len(graph)
    queue = deque([start])
    visited[start] = True

    while queue:
        node = queue.popleft()
        print(node, end=" ")

        for neighbor in graph[node]:
            if not visited[neighbor]:
                queue.append(neighbor)
                visited[neighbor] = True

def main():
    graph = [
        [1, 2],
        [0, 2, 3],
        [0, 1, 4],
        [1, 4],
        [2, 3]
    ]

    start_node = 0
    print("BFS traversal:", end=" ")
    bfs(graph, start_node)
    print()

main()
```



**Скорость**: временная сложность BFS составляет O(V + E), где V — количество вершин, а E — количество ребер в графе.

<a id="12"></a>

## Бинарный поиск

Двоичный поиск — это эффективный алгоритм поиска, который находит положение целевого значения в отсортированном массиве.

```cpp
#include <iostream>
#include <vector>

int binarySearch(const std::vector<int>& arr, int target) {
    int left = 0;
    int right = arr.size() - 1;
    
    while (left <= right) {
        int mid = left + (right - left) / 2;
        
        if (arr[mid] == target) {
            return mid;
        }
        
        if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
    
    return -1;
}

int main() {
    std::vector<int> arr = {1, 5, 8, 12, 15};
    int target = 8;
    int result = binarySearch(arr, target);
    
    if (result != -1) {
        std::cout << "Element found at index " << result << std::endl;
    } else {
        std::cout << "Element not found" << std::endl;
    }
    
    return 0;
}
```



swift

```swift
func binarySearch(_ arr: [Int], _ target: Int) -> Int {
    var left = 0
    var right = arr.count - 1
    
    while left <= right {
        let mid = left + (right - left) / 2
        
        if arr[mid] == target {
            return mid
        }
        
        if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    
    return -1
}

func main() {
    let arr = [1, 5, 8, 12, 15]
    let target = 8
    let result = binarySearch(arr, target)
    
    if result != -1 {
        print("Element found at index", result)
    } else {
        print("Element not found")
    }
}

main()
```



python:

```python
def binary_search(arr, target):
    left = 0
    right = len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            return mid

        if arr[mid] < target:
            left = mid + 1


        else:
            right = mid - 1

    return -1

def main():
    arr = [1, 5, 8, 12, 15]
    target = 8
    result = binary_search(arr, target)

    if result != -1:
        print("Element found at index", result)
    else:
        print("Element not found")

main()
```



**Скорость**: временная сложность бинарного поиска составляет O(log n), где n — количество элементов в отсортированном массиве.



## Строительство дерева

<a id="13"></a>

### Двоичное дерево поиска (BST)

Двоичное дерево поиска — это структура данных двоичного дерева, удовлетворяющая свойству бинарного поиска, в котором левый дочерний элемент узла меньше, а правый — больше.

```cpp
#include <iostream>

struct Node {
    int value;
    Node* left;
    Node* right;
    
    Node(int value) : value(value), left(nullptr), right(nullptr) {}
};

class BST {
private:
    Node* root;

    Node* insertRecursive(Node* node, int value) {
        if (node == nullptr) {
            return new Node(value);
        }

        if (value < node->value) {
            node->left = insertRecursive(node->left, value);
        } else if (value > node->value) {
            node->right = insertRecursive(node->right, value);
        }

        return node;
    }

public:
    BST() : root(nullptr) {}

    void insert(int value) {
        root = insertRecursive(root, value);
    }

    // Other methods for tree traversal, deletion, etc.
};

int main() {
    BST bst;
    bst.insert(5);
    bst.insert(2);
    bst.insert(8);
    bst.insert(12);
    bst.insert(1);

    // Other operations on the BST

    return 0;
}
```

swift

```swift
class Node {
    let value: Int
    var left: Node?
    var right: Node?
    
    init(_ value: Int) {
        self.value = value
        self.left = nil
        self.right = nil
    }
}

class BST {
    private var root: Node?
    
    private func insertRecursive(_ node: Node?, _ value: Int) -> Node {
        guard let node = node else {
            return Node(value)
        }
        
        if value < node.value {
            node.left = insertRecursive(node.left, value)
        } else if value > node.value {
            node.right = insertRecursive(node.right, value)
        }
        
        return node
    }
    
    func insert(_ value: Int) {
        root = insertRecursive(root, value)
    }
    
    // Other methods for tree traversal, deletion, etc.
}

func main() {
    let bst = BST()
    bst.insert(5)
    bst.insert(2)
    bst.insert(8)
    bst.insert(12)
    bst.insert(1)
    
    // Other operations on the BST
}

main()
```



python:

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, value):
        if self.root is None:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)

    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert_recursive(node.left, value)
        elif value > node.value:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert_recursive(node.right, value)

def main():
    bst = BST()
    bst.insert(5)
    bst.insert(2)
    bst.insert(8)
    bst.insert(12)
    bst.insert(1)

    # Other operations on the BST

main()
```

<a id="14"></a>

## Динамическое программирование

Динамическое программирование — это метод оптимизации, используемый для решения проблем путем их разбиения на перекрывающиеся подзадачи и сохранения результатов во избежание избыточных вычислений. Он часто обеспечивает значительное улучшение временной и пространственной сложности по сравнению с наивными подходами.

### Динамические переменные

В C++ динамические переменные создаются с помощью указателей и ключевого слова `new`. Они позволяют выделять память во время выполнения и могут быть изменены или освобождены по мере необходимости.

```cpp
int* dynamicInt = new int;
*dynamicInt = 10;

delete dynamicInt;  // Deallocate the memory
```



### Динамические массивы С++

Динамические массивы в C++ создаются с помощью указателей и выделяются с помощью ключевого слова `new`. Они позволяют изменять размер и освобождать память с помощью оператора `delete[]`.

```cpp
int size = 5;
int* dynamicArray = new int[size];

for (int i = 0; i < size; ++i) {
    dynamicArray[i] = i;
}

delete[] dynamicArray;  // Deallocate the memory
```

#### Динамические массивы swift

Динамические массивы могут быть созданы в Swift с использованием типа «Array». Размер массива можно изменять динамически, добавляя или удаляя элементы с помощью таких методов, как `append(_:)` и `remove(at:)`.

```swift
// Create an empty dynamic array
var dynamicArray = [Int]()

// Add elements to the array
dynamicArray.append(1)
dynamicArray.append(2)
dynamicArray.append(3)

// Remove an element from the array
dynamicArray.remove(at: 1)

// Access elements of the array
let firstElement = dynamicArray[0]
```

#### 

## Динамические массивы py

В Python динамические массивы могут быть реализованы с использованием структуры данных list. Списки автоматически изменяют свой размер при добавлении или удалении элементов. Вот пример:

```python
# Create an empty dynamic array
dynamic_array = []

# Append elements dynamically
dynamic_array.append(1)
dynamic_array.append(2)
dynamic_array.append(3)

# Access elements
print(dynamic_array[0])  # Output: 1
print(dynamic_array[1])  # Output: 2
print(dynamic_array[2])  # Output: 3
```



### Динамические классы

Динамические классы в C++ создаются с использованием указателей и выделяются с помощью ключевого слова `new`. Они обеспечивают гибкое управление памятью и могут быть освобождены с помощью оператора «удалить».

```cpp
class MyClass {
public:
    int value;
    
    MyClass(int v) : value(v) {}
    ~MyClass() {}
};

MyClass* dynamicObject = new MyClass(10);

delete dynamicObject;  // Deallocate the memory
```



#### Динамические классы

Динамические классы могут быть определены в Swift с помощью ключевого слова class. Эти классы могут иметь свойства и методы, которые можно динамически изменять и получать к ним доступ.

```swift
class DynamicClass {
    var dynamicProperty: Int

    init(dynamicProperty: Int) {
        self.dynamicProperty = dynamicProperty
    }

    func dynamicMethod() {
        // Perform some computation
    }
}

// Create an instance of the dynamic class
let dynamicObject = DynamicClass(dynamicProperty: 10)

// Modify the dynamic property
dynamicObject.dynamicProperty = 20

// Call the dynamic method
dynamicObject.dynamicMethod()
```



Для динамических классов вы можете использовать встроенное ключевое слово class в Python. Классы определяют схемы создания объектов с динамическими атрибутами и методами. Вот пример:

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        print(f"My name is {self.name} and I am {self.age} years old.")

# Create dynamic objects
person1 = Person("Alice", 25)
person2 = Person("Bob", 30)

# Access object attributes
print(person1.name)  # Output: Alice
print(person2.age)   # Output: 30

# Call object methods
person1.introduce()  # Output: My name is Alice and I am 25 years old.
person2.introduce()  # Output: My name is Bob and I am 30 years old.
```

В этом примере мы определяем класс «Person» с динамическими атрибутами «name» и «age», а также динамический метод «introduce()».



<a id="18"></a>

### Хранение нескольких классов

Для динамического хранения нескольких экземпляров класса вы можете использовать контейнеры, такие как векторы или динамически выделяемые массивы указателей классов.

```cpp
#include <iostream>
#include <vector>

class MyClass {
public:
    int value;
    
    MyClass(int v) : value(v) {}
};

int main() {
    std::vector<MyClass*> myClassVector;

    // Add objects to the vector
    myClassVector.push_back(new MyClass(10));
    myClassVector.push_back(new MyClass(20));
    myClassVector.push_back(new MyClass(30));

    // Access and use the objects
    for (const auto& obj : myClassVector) {
        std::cout << obj->value << std::endl;
    }

    // Deallocate the memory
    for (const auto& obj : myClassVector) {
        delete obj;
    }

    return 0;
}
```

#### Хранение нескольких классов

Чтобы хранить несколько классов в Swift, вы можете использовать коллекции, такие как массивы или словари. Например, вы можете создать массив для хранения экземпляров разных классов.

```swift
class ClassA {}
class ClassB {}

// Create an array to store instances of different classes
var classArray: [Any] = []

// Add instances to the array
classArray.append(ClassA())
classArray.append(ClassB())
```

<a id="19"></a>

## Правильное хранение нескольких классов

Чтобы правильно хранить несколько классов, вы можете использовать структуры данных, такие как списки или словари. Вот пример:

```python
class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade

class Teacher:
    def __init__(self, name, subject):
        self.name = name
        self.subject = subject

# Store multiple instances of classes
students = [
    Student("Alice", 10),
    Student("Bob", 11),
    Student("Carol", 9)
]

teachers = {
    "Math": Teacher("John", "Math"),
    "Science": Teacher("Jane", "Science")
}

# Access class instances
print(students[0].name)                  # Output: Alice
print(teachers["Math"].name)             # Output: John


print(teachers["Science"].subject)       # Output: Science
```

В этом примере мы определяем классы «Студент» и «Учитель» и сохраняем несколько их экземпляров в списке и словаре соответственно.

<a id="20"></a>

### Передача динамических объектов в функции

Чтобы передать динамические объекты функциям, вы можете использовать указатели или ссылки. Важно правильно управлять памятью, обеспечивая надлежащее освобождение, чтобы избежать утечек памяти.

```cpp
#include <iostream>

class MyClass {
public:
    int value;
    
    MyClass(int v) : value(v) {}
};

void processObject(MyClass* obj) {
    // Access and modify the object
    obj->value *= 2;
}

int main() {
    MyClass* dynamicObject = new MyClass(10);

    // Pass the dynamic object to a function
    processObject(dynamicObject);

    // Access and use the modified object
    std::cout << dynamicObject->value << std::endl;

    // Deallocate the memory
    delete dynamicObject;

    return 0;
}
```



#### Передача динамических объектов в функции

В Swift вы можете передавать динамические объекты функциям, используя тип Any или используя протокольно-ориентированное программирование и дженерики.

```swift
func processDynamicObject(object: Any) {
    // Perform operations on the dynamic object
}

// Call the function with a dynamic object
processDynamicObject(object: dynamicObject)
```

## Передача динамических объектов в функции

В Python объекты передаются по ссылке, поэтому изменения, внесенные в объекты внутри функции, повлияют на исходные объекты. Вот пример:

```python
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

def scale_rectangle(rect, factor):
    rect.width *= factor
    rect.height *= factor

# Create a dynamic rectangle object
rectangle = Rectangle(4, 3)

# Pass the object to the function
scale_rectangle(rectangle, 2)

# Access modified object attributes
print(rectangle.width)   # Output: 8
print(rectangle.height)  # Output: 6
```

В этом примере мы определяем класс `Rectangle` и функцию `scale_rectangle()`, которая изменяет ширину и высоту переданного ей прямоугольного объекта.

<a id="21"></a>

### Создание универсальных функций

Чтобы создать универсальные функции, которые могут принимать различные типы параметров, вы можете использовать шаблоны в C++. Шаблоны позволяют

  вам писать функции, которые параметризуются по типам.

```cpp
#include <iostream>

template <typename T>
void processObject(T* obj) {
    // Access and modify the object
    *obj *= 2;
}

int main() {
    int* dynamicInt = new int(10);
    double* dynamicDouble = new double(3.14);

    // Pass dynamic objects to the generic function
    processObject(dynamicInt);
    processObject(dynamicDouble);

    // Access and use the modified objects
    std::cout << *dynamicInt << std::endl;
    std::cout << *dynamicDouble << std::endl;

    // Deallocate the memory
    delete dynamicInt;
    delete dynamicDouble;

    return 0;
}
```



#### Общие функции

Чтобы создать функцию, которая не заботится о конкретном типе переданного параметра, вы можете использовать дженерики. Это позволяет функции работать с любым типом, удовлетворяющим определенным требованиям.

```swift
func processObject<T>(object: T) {
    // Perform operations on the object
}

// Call the function with different types of objects
processObject(object: 10)
processObject(object: "Hello")
```

## Гибкость типа параметра функции

В Python вы можете сделать функции гибкими, чтобы они могли принимать параметры любого типа, используя подсказку универсального типа `typing.Any`. Это позволяет функции обрабатывать параметры любого типа. Вот пример:

```python
from typing import Any

def process_parameter(parameter: Any) -> None:
    print(parameter)

# Call the function with different types of parameters
process_parameter(10)         # Output: 10
process_parameter("Hello")    # Output: Hello
process_parameter([1, 2, 3])  # Output: [1, 2, 3]
```

В этом примере функция process_parameter() принимает параметр типа Any и просто выводит его. Его можно вызывать с различными типами параметров, такими как целые числа, строки или списки.

Я надеюсь, что приведенные выше объяснения и код помогут вам понять динамическое программирование, динамические массивы и классы, хранение нескольких классов, передачу динамических объектов функциям и создание гибких функций в Python. Дайте мне знать, если у вас есть дополнительные вопросы!



## Алгоритмы обхода графа

<a id="22"></a>

### Алгоритм Дейкстры

Алгоритм Дейкстры используется для поиска кратчайшего пути во взвешенном графе от исходной вершины ко всем остальным вершинам.

```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <climits>

struct Edge {
    int destination;
    int weight;

    Edge(int dest, int w) : destination(dest), weight(w) {}
};

void Dijkstra(const std::vector<std::vector<Edge>>& graph, int source) {
    int numVertices = graph.size();
    std::vector<int> distance(numVertices, INT_MAX);
    std::vector<bool> visited(numVertices, false);
    distance[source] = 0;

    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<std::pair<int, int>>> pq;
    pq.push({0, source});

    while (!pq.empty()) {
        int currVertex = pq.top().second;
        pq.pop();
        visited[currVertex] = true;

        for (const Edge& edge : graph[currVertex]) {
            int newDistance = distance[currVertex] + edge.weight;
            if (!visited[edge.destination] && newDistance < distance[edge.destination]) {
                distance[edge.destination] = newDistance;
                pq.push({newDistance, edge.destination});
            }
        }
    }

    // Print the distances from the source vertex
    for (int i = 0; i < numVertices; ++i) {
        std::cout << "Distance from source to vertex " << i << ": " << distance[i] << std::endl;
    }
}

int main() {
    // Create the graph (adjacency list representation)
    int numVertices = 5;
    std::vector<std::vector<Edge>> graph(numVertices);

    // Add edges to the graph
    graph[0].push_back(Edge(1, 4));
    graph[0].push_back(Edge(2, 1));
    graph[1].push_back(Edge(3, 1));
    graph[2].push_back(Edge(1, 2));
    graph[2].push_back(Edge(3, 5));
    graph[3].push_back(Edge(4, 3));

    // Perform Dijkstra's algorithm
    int sourceVertex = 0;
    Dijkstra(graph, sourceVertex);

    return 0;
}
```



Алгоритм Дейкстры используется для поиска кратчайшего пути между узлами в графе с неотрицательными весами ребер.

```swift
import Foundation

struct Edge {
    let destination: Int
    let weight: Int
}

func dijkstraAlgorithm(graph: [[Edge]], source: Int) -> [Int] {
    let numVertices = graph.count
    var distance = Array(repeating: Int.max, count: numVertices)
    var visited = Array(repeating: false, count: numVertices)
    distance[source] = 0

    for _ in 0..<numVertices-1 {
        var minDistance = Int.max
        var minVertex = -1

        for v

 in 0..<numVertices {
            if !visited[v] && distance[v] < minDistance {
                minDistance = distance[v]
                minVertex = v
            }
        }

        visited[minVertex] = true

        for edge in graph[minVertex] {
            let newDistance = distance[minVertex] + edge.weight
            if newDistance < distance[edge.destination] {
                distance[edge.destination] = newDistance
            }
        }
    }

    return distance
}

// Main function
func main() {
    let numVertices = 5
    var graph: [[Edge]] = Array(repeating: [], count: numVertices)
    graph[0].append(Edge(destination: 1, weight: 4))
    graph[0].append(Edge(destination: 2, weight: 1))
    graph[1].append(Edge(destination: 3, weight: 1))
    graph[2].append(Edge(destination: 1, weight: 2))
    graph[2].append(Edge(destination: 3, weight: 5))
    graph[3].append(Edge(destination: 4, weight: 3))

    let source = 0
    let distances = dijkstraAlgorithm(graph: graph, source: source)
    print("Shortest distances from source \(source): \(distances)")
}

// Call the main function
main()
```

#### 



<a id="23"></a>

### Алгоритм Беллмана-Форда

Алгоритм Беллмана-Форда используется для поиска кратчайшего пути во взвешенном графе от исходной вершины ко всем другим вершинам, даже при наличии отрицательных весов ребер.

```cpp
#include <iostream>
#include <vector>
#include <climits>

struct Edge {
    int source;
    int destination;
    int weight;

    Edge(int src, int dest, int w) : source(src), destination(dest), weight(w) {}
};

void BellmanFord(const std::vector<Edge>&

 edges, int numVertices, int source) {
    std::vector<int> distance(numVertices, INT_MAX);
    distance[source] = 0;

    // Relax edges repeatedly
    for (int i = 0; i < numVertices - 1; ++i) {
        for (const Edge& edge : edges) {
            if (distance[edge.source] != INT_MAX && distance[edge.source] + edge.weight < distance[edge.destination]) {
                distance[edge.destination] = distance[edge.source] + edge.weight;
            }
        }
    }

    // Check for negative-weight cycles
    for (const Edge& edge : edges) {
        if (distance[edge.source] != INT_MAX && distance[edge.source] + edge.weight < distance[edge.destination]) {
            std::cout << "Graph contains a negative-weight cycle." << std::endl;
            return;
        }
    }

    // Print the distances from the source vertex
    for (int i = 0; i < numVertices; ++i) {
        std::cout << "Distance from source to vertex " << i << ": " << distance[i] << std::endl;
    }
}

int main() {
    // Create the graph (edge list representation)
    std::vector<Edge> edges = {
        Edge(0, 1, 4),
        Edge(0, 2, 1),
        Edge(1, 3, 1),
        Edge(2, 1, 2),
        Edge(2, 3, 5),
        Edge(3, 4, 3)
    };

    int numVertices = 5;
    int sourceVertex = 0;

    // Perform Bellman-Ford's algorithm
    BellmanFord(edges, numVertices, sourceVertex);

    return 0;
}
```





```swift
import Foundation

struct Edge {
    let source: Int
    let destination: Int
    let weight: Int
}

func bellmanFordAlgorithm(graph: [Edge], numVertices: Int, source: Int) -> [Int] {
    var distance = Array(repeating: Int.max, count: numVertices)
    distance[source] = 0

    for _ in 0..<numVertices-1 {
        for edge in graph {
            if distance[edge.source] != Int.max && distance[edge.source] + edge.weight < distance[edge.destination] {
                distance[edge.destination] = distance[edge.source] + edge.weight
            }
        }
    }

    for edge in graph {
        if distance[edge.source] != Int.max && distance[edge.source] + edge.weight < distance[edge.destination] {
            print("Graph contains a negative weight cycle")
            break
        }
    }

    return distance
}

// Main function
func main() {
    let numVertices = 5
    let graph = [
        Edge(source: 0, destination: 1, weight: 4),
        Edge(source: 0, destination: 2, weight: 1),
        Edge(source: 1, destination: 3, weight: 1),
        Edge(source: 2, destination: 1, weight: 2),
        Edge(source: 2, destination: 3, weight: 5),
        Edge(source: 3, destination: 4, weight: 3)
    ]

    let source = 0
    let distances = bellmanFordAlgorithm(graph: graph, numVertices: numVertices, source: source)
    print("Shortest distances from source \(source): \(distances)")
}

// Call the main function
main()
```

#### 

<a id="24"></a>

### Алгоритм Крускала

Алгоритм Крускала используется для нахождения минимального остовного дерева взвешенного графа.

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

struct Edge {
    int source;
    int destination;
    int weight;

    Edge(int src, int dest, int w) : source(src), destination(dest), weight(w) {}
};

bool sortByWeight(const Edge& e1, const Edge& e2) {
    return e1.weight < e2.weight;
}

class DisjointSet {
public:
    explicit DisjointSet(int size) : parent(size), rank(size, 0) {
        for (int i = 0; i < size; ++i) {
            parent[i] = i;
        }
    }

    int find(int vertex) {
        if (vertex != parent[vertex]) {
            parent[vertex] = find(parent[vertex]);
        }
        return parent[vertex];
    }

    void merge(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);

        if (rootX != rootY) {
            if (rank[rootX] < rank[rootY]) {
                parent[rootX] = rootY;
            } else if (rank[rootX] > rank[rootY]) {
                parent[rootY] = rootX;
            } else {
                parent[rootY] = rootX;
                rank[rootX]++;
            }
        }
    }

private:
    std::vector<int> parent;
    std::vector<int> rank;
};

void Kruskal(const std::vector<Edge>& edges, int numVertices) {
    std::vector<Edge> minimumSpanningTree;
    std::sort(edges.begin(), edges.end(), sortByWeight);
    DisjointSet disjointSet(numVertices);

    for (const Edge& edge : edges) {
        int sourceRoot = disjointSet.find(edge.source);
        int dest

Root = disjointSet.find(edge.destination);

        if (sourceRoot != destRoot) {
            minimumSpanningTree.push_back(edge);
            disjointSet.merge(sourceRoot, destRoot);
        }
    }

    // Print the minimum spanning tree
    for (const Edge& edge : minimumSpanningTree) {
        std::cout << edge.source << " - " << edge.destination << " (" << edge.weight << ")" << std::endl;
    }
}

int main() {
    // Create the graph (edge list representation)
    std::vector<Edge> edges = {
        Edge(0, 1, 4),
        Edge(0, 2, 1),
        Edge(1, 2, 2),
        Edge(1, 3, 1),
        Edge(2, 3, 5),
        Edge(2, 4, 3),
        Edge(3, 4, 4)
    };

    int numVertices = 5;

    // Perform Kruskal's algorithm
    Kruskal(edges, numVertices);

    return 0;
}
```





```swift
import Foundation

struct Edge {
    let source: Int
    let destination: Int
    let weight: Int
}

func kruskalAlgorithm(graph: [Edge], numVertices: Int) -> [Edge] {


    var parent = Array(0..<numVertices)
    var minimumSpanningTree: [Edge] = []

    func find(parent: inout [Int], vertex: Int) -> Int {
        if parent[vertex] != vertex {
            parent[vertex] = find(parent: &parent, vertex: parent[vertex])
        }
        return parent[vertex]
    }

    func union(parent: inout [Int], rank: inout [Int], vertex1: Int, vertex2: Int) {
        let root1 = find(parent: &parent, vertex: vertex1)
        let root2 = find(parent: &parent, vertex: vertex2)

        if rank[root1] < rank[root2] {
            parent[root1] = root2
        } else if rank[root1] > rank[root2] {
            parent[root2] = root1
        } else {
            parent[root2] = root1
            rank[root1] += 1
        }
    }

    let sortedEdges = graph.sorted { $0.weight < $1.weight }

    var i = 0
    var edgeCount = 0
    while edgeCount < numVertices - 1 {
        let currentEdge = sortedEdges[i]
        let root1 = find(parent: &parent, vertex: currentEdge.source)
        let root2 = find(parent: &parent, vertex: currentEdge.destination)

        if root1 != root2 {
            minimumSpanningTree.append(currentEdge)
            union(parent: &parent, rank: &rank, vertex1: root1, vertex2: root2)
            edgeCount += 1
        }

        i += 1
    }

    return minimumSpanningTree
}

// Main function
func main() {
    let numVertices = 5
    let graph = [
        Edge(source: 0, destination: 1, weight: 4),
        Edge(source: 0, destination: 2, weight: 1),
        Edge(source: 1, destination: 3, weight: 1),
        Edge(source: 2, destination: 1, weight: 2),
        Edge(source: 2, destination: 3, weight: 5),
        Edge(source: 3, destination: 4, weight: 3)
    ]

    let minimumSpanningTree = kruskalAlgorithm(graph: graph, numVertices: numVertices)
    print("Minimum Spanning Tree:")
    for edge in minimumSpanningTree {
        print("\(edge.source) - \(edge.destination) : \(edge.weight)")
    }
}

// Call the main function
main()
```

#### 

<a id="25"></a>

### Алгоритм Прима

Алгоритм Прима используется для нахождения минимального остовного дерева взвешенного графа.

```cpp
#include <iostream>
#include <vector>
#include <queue>
#include <climits>

struct Edge {
    int destination;
    int weight;

    Edge(int dest, int w) : destination(dest), weight(w) {}
};

void Prim(const std::vector<std::vector<Edge>>& graph, int numVertices) {
    std::vector<bool> visited(numVertices, false);
    std::vector<int> distance(numVertices, INT_MAX);
    std::vector<int> parent(numVertices, -1);
    distance[0] = 0;

    std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>, std::greater<std::pair<int, int>>> pq;
    pq.push({0, 0});

    while (!pq.empty()) {
        int currVertex = pq.top().second;
        pq.pop();
        visited[currVertex] = true;

        for (const Edge& edge : graph[currVertex]) {
            int newDistance = edge.weight;
            if (!visited[edge.destination] && newDistance < distance[edge.destination]) {
                distance[edge.destination] = newDistance;
                parent[edge.destination] = currVertex;
                pq.push({newDistance, edge.destination});
            }
        }
    }

    // Print the minimum spanning tree
    for (int i = 1; i < numVertices; ++i) {
        std::cout << parent[i] << " - " << i << " (" << distance[i] << ")" << std::endl;
    }
}

int main() {
    // Create the graph (adjacency list representation)
    int numVertices = 5;
    std::vector<std::vector<Edge>> graph(numVertices);

    // Add edges to the graph
    graph[0].push_back(Edge(1, 4));
    graph[0].push_back(Edge(2, 1));
    graph[1].push_back(Edge(0, 4));
    graph[1].push_back(Edge(2, 2));
    graph[1].push_back(Edge(3, 1));
    graph[2].push_back(Edge(0, 1));
    graph[2].push_back(Edge(1, 2));
    graph[2].push_back(Edge(3, 5));
    graph[2].push_back(Edge(4, 3));
    graph[3].push_back(Edge(1, 1));
    graph[3].push_back(Edge(2, 5));
    graph[3].push_back(Edge(4, 4));
    graph[4].push_back(Edge(2, 3));
    graph[4].push_back(Edge(3, 4));

    // Perform Prim's algorithm
    Prim(graph, numVertices);

    return 0;
}
```





```swift
import Foundation

struct Edge {
    let source: Int
    let destination: Int
    let weight: Int
}

func primAlgorithm(graph: [[(destination: Int, weight: Int)]]) -> [Edge] {
    let numVertices = graph.count
    var minimumSpanningTree: [Edge] = []
    var visited = Array(repeating: false, count: numVertices)
    var minHeap: [(destination: Int, weight: Int)] = []
    var parent: [Int?] = Array(repeating: nil, count: numVertices)

    func addToMinHeap(destination: Int, weight: Int) {
        minHeap.append((destination, weight))
        var currentIndex = minHeap.count - 1

        while currentIndex > 0 {
            let parentIndex = (currentIndex - 1) / 2

            if minHeap[parentIndex].weight > minHeap[currentIndex].weight {
                minHeap.swapAt(parentIndex, currentIndex)


                currentIndex = parentIndex
            } else {
                break
            }
        }
    }

    func extractMin() -> (destination: Int, weight: Int)? {
        guard !minHeap.isEmpty else { return nil }

        let minElement = minHeap[0]
        minHeap[0] = minHeap[minHeap.count - 1]
        minHeap.removeLast()

        var currentIndex = 0
        while true {
            let leftChildIndex = 2 * currentIndex + 1
            let rightChildIndex = 2 * currentIndex + 2

            var smallestIndex = currentIndex

            if leftChildIndex < minHeap.count && minHeap[leftChildIndex].weight < minHeap[smallestIndex].weight {
                smallestIndex = leftChildIndex
            }

            if rightChildIndex < minHeap.count && minHeap[rightChildIndex].weight < minHeap[smallestIndex].weight {
                smallestIndex = rightChildIndex
            }

            if smallestIndex != currentIndex {
                minHeap.swapAt(currentIndex, smallestIndex)
                currentIndex = smallestIndex
            } else {
                break
            }
        }

        return minElement
    }

    addToMinHeap(destination: 0, weight: 0)

    while let minElement = extractMin() {
        let currentVertex = minElement.destination
        visited[currentVertex] = true

        if let p = parent[currentVertex] {
            minimumSpanningTree.append(Edge(source: p, destination: currentVertex, weight: minElement.weight))
        }

        for edge in graph[currentVertex] {
            let destination = edge.destination
            let weight = edge.weight

            if !visited[destination] {
                addToMinHeap(destination: destination, weight: weight)
                parent[destination] = currentVertex
            }
        }
    }

    return minimumSpanningTree
}

// Main function
func main() {
    let numVertices = 5
    let graph: [[(destination: Int, weight: Int)]] = [
        [(1, 4), (2, 1)],
        [(0, 4), (3, 1)],
        [(0, 1), (3, 5)],
        [(1, 1), (2, 5), (4, 3)],
        [(3, 3)]
    ]

    let minimumSpanningTree = primAlgorithm(graph: graph)
    print("Minimum Spanning Tree:")
    for edge in minimumSpanningTree {
        print("\(edge.source) - \(edge.destination) : \(edge.weight)")
    }
}

// Call the main function
main()
```



