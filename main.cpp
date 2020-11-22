// Imports the necessary libraries
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>
#include <time.h>
#include <unistd.h>

using namespace std;
// Sets max number of iterations and the thinking and eating states
#define MAX_ITERATIONS 10
#define THINKING 0
#define EATING 1
// Error checking macro
#define ERRCHECK(a) do {                                                \
    int ret = (a);                                                      \
    if (ret != 0){                                                      \
      cerr << "Error in function " << #a << " at " << __FILE__ << ":" << __LINE__ << endl; \
      return 1;                                                           \
    }                                                                   \
  } while (0)

/**
    Timing function found in professor Kaeli's examples.

    @return current time in ms.
*/
double CLOCK() {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC,  &t);
  return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

int num_philosophers;
omp_lock_t *forks;
int *philosopher_state;

// Struct containing info about thread's task
typedef struct thread_info{
  int left;
  int right;
} thread_info_t;

/**
    Debugging function for printing an array of any type (using templates).

    @param arr: array to print.
    @param size: size of array.
*/
template <class T>
void print_arr(T *arr, int size){
  // Debug function for printing array

  for (int i = 0; i < size; i++){
    cout << arr[i] << " ";
  }
  cout << endl;
}

/**
    Proceed to think using a constant time.

    @param tid: thread id of thread eating.
*/
void think(int tid){
#ifdef DEBUG
  pthread_mutex_lock(&print_lock);
  cout << "Philosopher #" << tid << " thinking!" << endl;
  pthread_mutex_unlock(&print_lock);
#endif
  philosopher_state[tid] = THINKING;
  usleep(100000);
}

/**
    Proceed to eat using a constant time.

    @param tid: thread id of thread eating.
*/
void eat(int tid){
#ifdef DEBUG
  pthread_mutex_lock(&print_lock);
  cout << "Philosopher #" << tid << " eating!" << endl;
  pthread_mutex_unlock(&print_lock);
#endif
  philosopher_state[tid] = EATING;
  usleep(100000);
}

/**
    Pick up both forks.

    @param thread_info: Struct containing information about the thread's task.
*/
bool grab_fork(thread_info_t *thread_info){
  int left_fork = thread_info->left;
  int right_fork = thread_info->right;
  int tid = omp_get_thread_num();

  // Pick up left fork and attempt to pick up right fork
  // If right fork can't be picked up then drop left fork
  // and try again until successful picking up both forks
  while(true){
    // Acquire left fork
    omp_set_lock(&forks[left_fork]);

    // Use a nonblocking trylock to attempt to pick up right fork
    // If successful, breakout of infinite loop otherwise return false
    if (omp_test_lock(&forks[right_fork]) != 0){
      break;
    }
    else{
      omp_unset_lock(&forks[left_fork]);
      return false;
    }
  }

  // Proceed to eat
  eat(tid);
  return true;
}

/**
    Put down both forks.

    @param thread_info: Struct containing information about the thread's task.
*/
void putdown_fork(thread_info_t *thread_info){
  int left_fork = thread_info->left;
  int right_fork = thread_info->right;
  int tid = omp_get_thread_num();
  omp_unset_lock(&forks[left_fork]);
  omp_unset_lock(&forks[right_fork]);
  think(tid);
}

/**
    Prints out the state using just the master thread.

    @param thread_info: Struct containing information about the thread's task.
    @return NULL.
*/
void print_state(int iteration){
#pragma omp master
  {
    cout << "Iteration: " << iteration << endl;
    print_arr<int>(philosopher_state, num_philosophers);
  }
}

/**
    Multithreaded philosopher simulation.

    @param thread_info: Struct containing information about the thread's task.
    @return NULL.
*/
void philosopher(thread_info_t *thread_info){
  // Retrieves the current running thread's thread id
  int tid = omp_get_thread_num();
  // Initialize variable to store the iteration
  int iteration = 0;
  bool got_fork = false;
  // Iterate until the iteration reaches the maximum number
  // of iterations
  while (iteration < MAX_ITERATIONS){
    // attempts to grab fork, function returns true
    // if forks were successfully picked up and false
    // otherwise
    got_fork = grab_fork(thread_info);
    // Have 1 thread print out the state
    print_state(iteration);
    // If the forks were picked up, put them down in this step
    if (got_fork)
      putdown_fork(thread_info);
    // If the forks were not picked up, have philosopher think
    if (!got_fork)
      think(tid);
    // Increments the iteration number by 1
    ++iteration;
  }
}

/**
    Controls the flow of the program execution.
*/
int main(int argc, char *argv[]){
  // Exits program if the number of philosophers is not added in as a
  // command line argument
  if (argc != 2) {
    cout << "Usage: " << argv[0] << " num_philosophers" << endl;
    return 1;
  }
  num_philosophers = atoi(argv[1]);

  // Initial print statement to print out information about the number
  // of philosophers the program is running with
  printf("Running simulation with %d philosophers\n", num_philosophers);

  // Dynamically allocate forks (omp locks) and philosopher state
  // to store information about the state of the philosophers
  forks = new omp_lock_t[num_philosophers];
  philosopher_state = new int[num_philosophers];
  memset(philosopher_state, THINKING, (num_philosophers) * sizeof(int));

  // Initialize omp locks
  for (int i = 0; i < num_philosophers; ++i)
    omp_init_lock(&forks[i]);

  // Initialize the array of thread_info_t structs
  thread_info_t thread_info[num_philosophers];

  // Initialize the structs in the array with the correct information
  // storing the appropriate information about each struct
  for (int i = 0; i < num_philosophers; ++i){
    thread_info[i].left = i;
    thread_info[i].right = (i + 1) % num_philosophers;
  }

  // Initialize values to store time
  double start, finish;
  // Get initial time before the parallel subset of the program
  // gets executed
  start = CLOCK();

  // Use omp pragma directive to run a subset of the program in parallel
  // Number of threads is determined by the number of threads on the system
#pragma omp parallel num_threads(num_philosophers) shared(philosopher_state)
  {
    // retrieves the thread id to retrieve the correct struct in the array
    int tid = omp_get_thread_num();
    // Simulates dining philosopher problem
    philosopher(&thread_info[tid]);
  }

  // Finish timing process by retrieving the current time in ms
  finish = CLOCK();

  // Outputs the time taken to execute the parallel subset of the program
  cout << "Time taken: " << finish - start << endl;

  // Destroy omp locks
  for (int i = 0; i < num_philosophers; ++i){
    omp_destroy_lock(&forks[i]);
  }

  // Free up memory
  delete[] forks;
  delete[] philosopher_state;
  return 0;
}
