#include "types.h"
#include "param.h"
#include "memlayout.h"
#include "riscv.h"
#include "spinlock.h"
#include "proc.h"
#include "defs.h"
#include <stdbool.h>

//-----------------------------------------------------

#ifdef ON
int blnc_flag = 1;
#else
int blnc_flag = 0;
#endif

//-----------------------------------------------------
uint64 MAX_UINT64 = 94467440737095515;



int number_of_cpus;
struct cpu cpus[NCPU];

struct proc proc[NPROC];

struct proc *initproc;

int nextpid = 1;
struct spinlock pid_lock;

extern void forkret(void);
static void freeproc(struct proc *p);

extern char trampoline[]; // trampoline.S

// helps ensure that wakeups of wait()ing
// parents are not lost. helps obey the
// memory model when using p->parent.
// must be acquired before any p->lock.
struct spinlock wait_lock;

extern uint64 cas(volatile void *addr, int expected, int newval);
int UNUSED_list_head = -1;
int SLEEPING_list_head = -1;
int ZOMBIE_list_head = -1;
struct spinlock UNUSED_list_head_lock;
struct spinlock SLEEPING_list_head_lock;
struct spinlock ZOMBIE_list_head_lock;
//-----------------------------------------------------------------------------
// adding the new elment to the tail of the list
void List_insert(int *list_head_index, int index_of_process_to_add, struct spinlock *list_lock)
{

  acquire(list_lock);
  bool empty_list_flag = false;

  struct proc *process_to_add = &proc[index_of_process_to_add];
  if (*list_head_index == -1)//the list is empty
  {
    *list_head_index = process_to_add->index_in_proc_array;
    process_to_add->next_pid = -1;
    empty_list_flag = true;
  }
  release(list_lock);
  if (empty_list_flag)
  {
    return;
  }

  struct proc *p = &proc[*list_head_index];
  // p = &proc[proc_list_head->pid];
  acquire(&p->lock_for_list_operations);
  while (p->next_pid != -1)
  {
    release(&p->lock_for_list_operations);
    p = &proc[p->next_pid];
    acquire(&p->lock_for_list_operations);
  }
  p->next_pid = index_of_process_to_add;
  process_to_add->next_pid = -1;
  release(&p->lock_for_list_operations);
}

bool List_remove(int *list_head_index, int index_of_proc_to_remove, struct spinlock *list_lock)
{
  acquire(list_lock);
  if (*list_head_index == -1) // empty list
  {
    release(list_lock);
    return false;
  }
  release(list_lock);

  struct proc *p;

  acquire(list_lock);
  if (*list_head_index == index_of_proc_to_remove)
  {

    p = &proc[*list_head_index];
    // p = &proc[list_head->pid];

    acquire(&p->lock_for_list_operations);
    *list_head_index = p->next_pid;
    release(&p->lock_for_list_operations);

    release(list_lock);
    return true;
  }
  release(list_lock);

  bool proc_not_in_the_list = false;

  struct proc *prev = &proc[*list_head_index];
  acquire(&prev->lock_for_list_operations);
  p = &proc[prev->next_pid];

  acquire(&p->lock_for_list_operations);

  bool stop = false;
  while (!stop)
  {

    if (prev->next_pid == -1) // end of the list
    {
      stop = true;
      proc_not_in_the_list = true;
      continue;
    }

    if (p->index_in_proc_array == index_of_proc_to_remove)
    {
      stop = true;
      prev->next_pid = p->next_pid;
      continue;
    }
    release(&prev->lock_for_list_operations);

    prev = p;
    p = &proc[p->next_pid];
    acquire(&p->lock_for_list_operations);
  }
  release(&prev->lock_for_list_operations);
  release(&p->lock_for_list_operations);
  if (proc_not_in_the_list)
  {
    return false;
  }
  else
  {
    return true;
  }
}

int set_CPU(int cpu_num)
{
  struct proc *p = myproc();
  if (!cas(&p->process_cpu_index, p->process_cpu_index, cpu_num))
  {
    return -1;
  }
  yield();
  return cpu_num;
}
// initate the cpu's list
void init_CPU_RUNNABLE_list()
{
  struct cpu *loop_var;
  for (loop_var = cpus; loop_var < &cpus[NCPU]; loop_var++)
  {
    loop_var->RUNNABLE_list_head_pid = -1;
  }
}
//-----------------------------------------------------------------------------

// Allocate a page for each process's kernel stack.
// Map it high in memory, followed by an invalid
// guard page.
void proc_mapstacks(pagetable_t kpgtbl)
{
  struct proc *p;

  for (p = proc; p < &proc[NPROC]; p++)
  {
    char *pa = kalloc();
    if (pa == 0)
      panic("kalloc");
    uint64 va = KSTACK((int)(p - proc));
    kvmmap(kpgtbl, va, (uint64)pa, PGSIZE, PTE_R | PTE_W);
  }
}

// initialize the proc table at boot time.
void procinit(void)
{
  struct proc *p;
  int index_for_proecess = 0;

  initlock(&pid_lock, "nextpid");
  initlock(&wait_lock, "wait_lock");
  
  acquire(&UNUSED_list_head_lock);
  for (p = proc; p < &proc[NPROC]; p++)
  {
    initlock(&p->lock, "proc");
    initlock(&p->lock_for_list_operations, "lock_for_list_operations");
    p->next_pid = index_for_proecess + 1;
    if (index_for_proecess == NPROC - 1) // max num of processs
    {
      p->next_pid = -1;
    }

    p->kstack = KSTACK((int)(p - proc));
    index_for_proecess = index_for_proecess + 1;
  }
  UNUSED_list_head = 0;
  release(&UNUSED_list_head_lock);
}

// Must be called with interrupts disabled,
// to prevent race with process being moved
// to a different CPU.
int cpuid()
{
  int id = r_tp();
  return id;
}

// Return this CPU's cpu struct.
// Interrupts must be disabled.
struct cpu *
mycpu(void)
{
  int id = cpuid();
  struct cpu *c = &cpus[id];
  return c;
}

// Return the current struct proc *, or zero if none.
struct proc *
myproc(void)
{
  push_off();
  struct cpu *c = mycpu();
  struct proc *p = c->proc;
  pop_off();
  return p;
}

int allocpid()
{
  int pid;
  do
  {
    pid = nextpid;

  } while (cas(&nextpid, pid, pid + 1));
  return pid;
}

// Look in the process table for an UNUSED proc.
// If found, initialize state required to run in the kernel,
// and return with p->lock held.
// If there are no free procs, or a memory allocation fails, return 0.
static struct proc *
allocproc(void)
{
  struct proc *p;
  //----------------------------------
  int index_for_proc_array = 0;
  for (p = proc; p < &proc[NPROC]; p++)
  {
    acquire(&p->lock);
    if (p->state == UNUSED)
    {
      goto found;
    }
    else
    {
      release(&p->lock);
    }
    index_for_proc_array = index_for_proc_array + 1;
  }
  return 0;
  //----------------------------------
found:
  p->pid = allocpid();

  p->state = USED;
  p->next_pid = -1;
  p->process_cpu_index = cpuid();
  p->index_in_proc_array = index_for_proc_array;
  List_remove(&UNUSED_list_head, p->index_in_proc_array, &UNUSED_list_head_lock);
  //  Allocate a trapframe page.
  if ((p->trapframe = (struct trapframe *)kalloc()) == 0)
  {
    freeproc(p);
    release(&p->lock);
    return 0;
  }
  // An empty user page table.
  p->pagetable = proc_pagetable(p);
  if (p->pagetable == 0)
  {
    freeproc(p);
    release(&p->lock);
    return 0;
  }
  // Set up new context to start executing at forkret,
  // which returns to user space.
  memset(&p->context, 0, sizeof(p->context));
  p->context.ra = (uint64)forkret;
  p->context.sp = p->kstack + PGSIZE;
  return p;
}

// free a proc structure and the data hanging from it,
// including user pages.
// p->lock must be held.
static void
freeproc(struct proc *p)
{
  if (p->trapframe)
    kfree((void *)p->trapframe);
  p->trapframe = 0;
  if (p->pagetable)
    proc_freepagetable(p->pagetable, p->sz);

  List_remove(&ZOMBIE_list_head, p->index_in_proc_array, &ZOMBIE_list_head_lock);
  p->pagetable = 0;
  p->sz = 0;
  p->pid = 0;
  p->parent = 0;
  p->name[0] = 0;
  p->chan = 0;
  p->killed = 0;
  p->xstate = 0;
  p->state = UNUSED;

  List_insert(&UNUSED_list_head, p->index_in_proc_array, &UNUSED_list_head_lock);
}

// Create a user page table for a given process,
// with no user memory, but with trampoline pages.
pagetable_t
proc_pagetable(struct proc *p)
{
  pagetable_t pagetable;

  // An empty page table.
  pagetable = uvmcreate();
  if (pagetable == 0)
    return 0;

  // map the trampoline code (for system call return)
  // at the highest user virtual address.
  // only the supervisor uses it, on the way
  // to/from user space, so not PTE_U.
  if (mappages(pagetable, TRAMPOLINE, PGSIZE,
               (uint64)trampoline, PTE_R | PTE_X) < 0)
  {
    uvmfree(pagetable, 0);
    return 0;
  }

  // map the trapframe just below TRAMPOLINE, for trampoline.S.
  if (mappages(pagetable, TRAPFRAME, PGSIZE,
               (uint64)(p->trapframe), PTE_R | PTE_W) < 0)
  {
    uvmunmap(pagetable, TRAMPOLINE, 1, 0);
    uvmfree(pagetable, 0);
    return 0;
  }

  return pagetable;
}

// Free a process's page table, and free the
// physical memory it refers to.
void proc_freepagetable(pagetable_t pagetable, uint64 sz)
{
  uvmunmap(pagetable, TRAMPOLINE, 1, 0);
  uvmunmap(pagetable, TRAPFRAME, 1, 0);
  uvmfree(pagetable, sz);
}

// a user program that calls exec("/init")
// od -t xC initcode
uchar initcode[] = {
    0x17, 0x05, 0x00, 0x00, 0x13, 0x05, 0x45, 0x02,
    0x97, 0x05, 0x00, 0x00, 0x93, 0x85, 0x35, 0x02,
    0x93, 0x08, 0x70, 0x00, 0x73, 0x00, 0x00, 0x00,
    0x93, 0x08, 0x20, 0x00, 0x73, 0x00, 0x00, 0x00,
    0xef, 0xf0, 0x9f, 0xff, 0x2f, 0x69, 0x6e, 0x69,
    0x74, 0x00, 0x00, 0x24, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00};

// Set up first user process.
void userinit(void)
{

  
struct cpu *loop_var;
// cpu's init
  for (loop_var = cpus; loop_var < &cpus[NCPU]; loop_var++)
  {
    loop_var->RUNNABLE_list_head_pid = -1;
  }
  //-----------------------------------------------------------
  struct proc *p;
  p = allocproc();
  initproc = p;

  // struct cpu *c = mycpu();

  // allocate one user page and copy init's instructions
  // and data into it.
  uvminit(p->pagetable, initcode, sizeof(initcode));
  p->sz = PGSIZE;

  // prepare for the very first "return" from kernel to user.
  p->trapframe->epc = 0;     // user program counter
  p->trapframe->sp = PGSIZE; // user stack pointer

  safestrcpy(p->name, "initcode", sizeof(p->name));
  p->cwd = namei("/");

  p->state = RUNNABLE;

  struct cpu *c = mycpu();
  // int CPU_runnable_list_head_index_in_proc_array = c->RUNNABLE_list_head_pid;
  List_insert(&c->RUNNABLE_list_head_pid, p->index_in_proc_array, &c->CPU_proc_list_lock); // admit the init process to the first CPU's list

  release(&p->lock);
}

// Grow or shrink user memory by n bytes.
// Return 0 on success, -1 on failure.
int growproc(int n)
{
  uint sz;
  struct proc *p = myproc();

  sz = p->sz;
  if (n > 0)
  {
    if ((sz = uvmalloc(p->pagetable, sz, sz + n)) == 0)
    {
      return -1;
    }
  }
  else if (n < 0)
  {
    sz = uvmdealloc(p->pagetable, sz, sz + n);
  }
  p->sz = sz;
  return 0;
}

// Create a new process, copying the parent.
// Sets up child kernel stack to return as if from fork() system call.
int fork(void)
{
  int i, pid;
  struct proc *np; // new process
  struct proc *p = myproc();

  // Allocate process.
  if ((np = allocproc()) == 0)
  {
    return -1;
  }

  // Copy user memory from parent to child.
  if (uvmcopy(p->pagetable, np->pagetable, p->sz) < 0)
  {
    freeproc(np);
    release(&np->lock);
    return -1;
  }
  np->sz = p->sz;

  // copy saved user registers.
  *(np->trapframe) = *(p->trapframe);

  // Cause fork to return 0 in the child.
  np->trapframe->a0 = 0;

  // increment reference counts on open file descriptors.
  for (i = 0; i < NOFILE; i++)
    if (p->ofile[i])
      np->ofile[i] = filedup(p->ofile[i]);
  np->cwd = idup(p->cwd);

  safestrcpy(np->name, p->name, sizeof(p->name));

  pid = np->pid;

  release(&np->lock);

  acquire(&wait_lock);
  //-------------------------------------------------
  int cpu_idindex = p->process_cpu_index;
  if (blnc_flag)
  { 
    
    int min = cpus[0].admitted_counter;
    int index = 0;
    int loop_var = 1;
    while (loop_var < number_of_cpus)
    {
      if (min > cpus[loop_var].admitted_counter)
      {
        index = loop_var;
        min = cpus[loop_var].admitted_counter;
      }
      loop_var++;
    }

    cpu_idindex = index;
  }
  while (!cas(&cpus[cpu_idindex].admitted_counter, *&cpus[cpu_idindex].admitted_counter, *&cpus[cpu_idindex].admitted_counter + 1) == 0)
    ;
  //-------------------------------------------------
  np->process_cpu_index = cpu_idindex;
  np->parent = p;
  release(&wait_lock);
  //-------------------------problems from here
  acquire(&np->lock);
  np->state = RUNNABLE;

  // int *pointer_to_process_cpuIndex = &cpus[p->process_cpu_index].RUNNABLE_list_head_pid;
  struct cpu *process_cpu = &cpus[p->process_cpu_index];

  List_insert(&process_cpu->RUNNABLE_list_head_pid, np->index_in_proc_array, &process_cpu->CPU_proc_list_lock);
  release(&np->lock);
  return pid;
}

// Pass p's abandoned children to init.
// Caller must hold wait_lock.
void reparent(struct proc *p)
{
  struct proc *pp;

  for (pp = proc; pp < &proc[NPROC]; pp++)
  {
    if (pp->parent == p)
    {
      pp->parent = initproc;
      wakeup(initproc);
    }
  }
}

// Exit the current process.  Does not return.
// An exited process remains in the zombie state
// until its parent calls wait().
void exit(int status)
{
  struct proc *p = myproc();

  if (p == initproc)
  {
    panic("init exiting");
  }

  // Close all open files.
  for (int fd = 0; fd < NOFILE; fd++)
  {
    if (p->ofile[fd])
    {
      struct file *f = p->ofile[fd];
      fileclose(f);
      p->ofile[fd] = 0;
    }
  }

  begin_op();
  iput(p->cwd);
  end_op();
  p->cwd = 0;

  acquire(&wait_lock);

  // Give any children to init.
  reparent(p);

  // Parent might be sleeping in wait().
  wakeup(p->parent);

  acquire(&p->lock);

  p->xstate = status;
  p->state = ZOMBIE;
  //---i added
  List_insert(&ZOMBIE_list_head, p->index_in_proc_array, &ZOMBIE_list_head_lock);
  //----------
  release(&wait_lock);

  // Jump into the scheduler, never to return.
  sched();
  panic("zombie exit");
}

// Wait for a child process to exit and return its pid.
// Return -1 if this process has no children.
int wait(uint64 addr)
{
  struct proc *np;
  int havekids, pid;
  struct proc *p = myproc();

  acquire(&wait_lock);

  for (;;)
  {
    // Scan through table looking for exited children.
    havekids = 0;
    for (np = proc; np < &proc[NPROC]; np++)
    {
      if (np->parent == p)
      {
        // make sure the child isn't still in exit() or swtch().
        acquire(&np->lock);

        havekids = 1;
        if (np->state == ZOMBIE)
        {
          // Found one.
          pid = np->pid;
          if (addr != 0 && copyout(p->pagetable, addr, (char *)&np->xstate,
                                   sizeof(np->xstate)) < 0)
          {
            release(&np->lock);
            release(&wait_lock);
            return -1;
          }
          freeproc(np);
          release(&np->lock);
          release(&wait_lock);
          return pid;
        }
        release(&np->lock);
      }
    }

    // No point waiting if we don't have any children.
    if (!havekids || p->killed)
    {
      release(&wait_lock);
      return -1;
    }

    // Wait for a child to exit.
    sleep(p, &wait_lock); // DOC: wait-sleep
  }
}

// Per-CPU process scheduler.
// Each CPU calls scheduler() after setting itself up.
// Scheduler never returns.  It loops, doing:
//  - choose a process to run.
//  - swtch to start running that process.
//  - eventually that process transfers control
//    via swtch back to the scheduler.
void scheduler(void)
{
  struct proc *p;
  // struct proc *loop_var;
  struct cpu *c = mycpu();
  c->proc = 0;

  for (;;)
  {
    // Avoid deadlock by ensuring that devices can interrupt.
    intr_on();

    while (c->RUNNABLE_list_head_pid != -1) // i changed the loop to iterate over CPU's list
    {
      p = &proc[c->RUNNABLE_list_head_pid];
      acquire(&p->lock);
      // if (p->state == RUNNABLE)
      //{
      //  Switch to chosen process.  It is the process's job
      //  to release its lock and then reacquire it
      //  before jumping back to us.

      List_remove(&c->RUNNABLE_list_head_pid, p->index_in_proc_array, &c->CPU_proc_list_lock);
      p->state = RUNNING;
      c->proc = p;

      swtch(&c->context, &p->context);
      // Process is done running for now.
      // It should have changed its p->state before coming back.
      c->proc = 0;
      release(&p->lock);
    }
  }
}

// Switch to scheduler.  Must hold only p->lock
// and have changed proc->state. Saves and restores
// intena because intena is a property of this
// kernel thread, not this CPU. It should
// be proc->intena and proc->noff, but that would
// break in the few places where a lock is held but
// there's no process.
void sched(void)
{
  int intena;
  struct proc *p = myproc();

  if (!holding(&p->lock))
    panic("sched p->lock");
  if (mycpu()->noff != 1)
    panic("sched locks");
  if (p->state == RUNNING)
    panic("sched running");
  if (intr_get())
    panic("sched interruptible");

  intena = mycpu()->intena;
  swtch(&p->context, &mycpu()->context);
  mycpu()->intena = intena;
}

// Give up the CPU for one scheduling round.
void yield(void)
{
  // struct cpu *c = mycpu();

  struct proc *p = myproc();
  acquire(&p->lock);
  p->state = RUNNABLE;
  struct cpu *cpu_process = &cpus[p->process_cpu_index];
  List_insert(&cpu_process->RUNNABLE_list_head_pid, p->index_in_proc_array, &cpu_process->CPU_proc_list_lock);

  sched();
  release(&p->lock);
}

// A fork child's very first scheduling by scheduler()
// will swtch to forkret.
void forkret(void)
{
  static int first = 1;

  // Still holding p->lock from scheduler.
  release(&myproc()->lock);

  if (first)
  {
    // File system initialization must be run in the context of a
    // regular process (e.g., because it calls sleep), and thus cannot
    // be run from main().
    first = 0;
    fsinit(ROOTDEV);
  }

  usertrapret();
}

// Atomically release lock and sleep on chan.
// Reacquires lock when awakened.
void sleep(void *chan, struct spinlock *lk)
{
  struct proc *p = myproc();

  // Must acquire p->lock in order to
  // change p->state and then call sched.
  // Once we hold p->lock, we can be
  // guaranteed that we won't miss any wakeup
  // (wakeup locks p->lock),
  // so it's okay to release lk.

  acquire(&p->lock); // DOC: sleeplock1
  //----i added----------------------------------
  List_insert(&SLEEPING_list_head, p->index_in_proc_array, &SLEEPING_list_head_lock);
  //---------------------------------------------

  release(lk);

  // Go to sleep.
  p->chan = chan;
  p->state = SLEEPING;

  sched();

  // Tidy up.
  p->chan = 0;

  // Reacquire original lock.
  release(&p->lock);
  acquire(lk);
}

// Wake up all processes sleeping on chan.
// Must be called without any p->lock.
void wakeup(void *chan)
{
  // no sleeping process
  acquire(&SLEEPING_list_head_lock);
  if (SLEEPING_list_head == -1)
  {
    release(&SLEEPING_list_head_lock);
    return;
  }
  struct cpu *c;
  struct proc *loop_var = &proc[SLEEPING_list_head];
  // loop_var = &proc[SLEEPING_list_head];
  release(&SLEEPING_list_head_lock);
  acquire(&loop_var->lock);
  int current_id = loop_var->index_in_proc_array;

  bool removed = false;
  release(&loop_var->lock);

  while (current_id != -1)
  {
    loop_var = &proc[current_id];
    acquire(&loop_var->lock);
    if (loop_var->state == SLEEPING && loop_var->chan == chan)
    {
      removed = List_remove(&SLEEPING_list_head, current_id, &SLEEPING_list_head_lock);
      if (removed)
      {
        loop_var->state = RUNNABLE;
        //--------------------------------------------
        int cpu_idindex = loop_var->process_cpu_index;
        if (blnc_flag)
        { // choose different cpu
          int index = 0;
          int min = cpus[0].admitted_counter;
          int i = 1;
          while (i < number_of_cpus)
          {
            if (min > cpus[i].admitted_counter)
            {
              index = i;
              min = cpus[i].admitted_counter;
            }
            i++;
          }
          cpu_idindex = index;
        }
        if (cpu_idindex != loop_var->process_cpu_index)
        {
          while (!cas(&cpus[cpu_idindex].admitted_counter, *&cpus[cpu_idindex].admitted_counter, *&cpus[cpu_idindex].admitted_counter + 1) == 0);
        }

        //--------------------------------------------
        loop_var->process_cpu_index = cpu_idindex;
        // loop_var->process_cpu_index = set_process_to_correct_cpu(loop_var->process_cpu_index, false);
        c = &cpus[loop_var->process_cpu_index];

        // int *process_cpuIndex = &cpus[loop_var->process_cpu_index].RUNNABLE_list_head_pid;
        // struct cpu *process_cpu = &cpus[loop_var->process_cpu_index];
        List_insert(&c->RUNNABLE_list_head_pid, current_id, &c->CPU_proc_list_lock);
      }
    }
    current_id = loop_var->next_pid;
    release(&loop_var->lock);
  }
}

// Kill the process with the given pid.
// The victim won't exit until it tries to return
// to user space (see usertrap() in trap.c).
int kill(int pid)
{
  struct proc *p;
  bool removed_proc = false;

  for (p = proc; p < &proc[NPROC]; p++)
  {
    acquire(&p->lock);
    if (p->pid == pid)
    {
      p->killed = 1;
      if (p->state == SLEEPING) // Wake process from sleep().
      {
        removed_proc = List_remove(&SLEEPING_list_head, p->index_in_proc_array, &SLEEPING_list_head_lock);
        if (removed_proc)
        {
          p->state = RUNNABLE;

          struct cpu *c = &cpus[p->process_cpu_index];
          List_insert(&c->RUNNABLE_list_head_pid, p->index_in_proc_array, &c->CPU_proc_list_lock);
        }
      }
      release(&p->lock);
      return 0;
    }
    release(&p->lock);
  }
  return -1;
}

int cpu_process_count(int cpu_num)
{
  struct cpu *selected_cpu = &cpus[cpu_num];
  return selected_cpu->admitted_counter;
}

// Copy to either a user address, or kernel address,
// depending on usr_dst.
// Returns 0 on success, -1 on error.
int either_copyout(int user_dst, uint64 dst, void *src, uint64 len)
{
  struct proc *p = myproc();
  if (user_dst)
  {
    return copyout(p->pagetable, dst, src, len);
  }
  else
  {
    memmove((char *)dst, src, len);
    return 0;
  }
}

// Copy from either a user address, or kernel address,
// depending on usr_src.
// Returns 0 on success, -1 on error.
int either_copyin(void *dst, int user_src, uint64 src, uint64 len)
{
  struct proc *p = myproc();
  if (user_src)
  {
    return copyin(p->pagetable, dst, src, len);
  }
  else
  {
    memmove(dst, (char *)src, len);
    return 0;
  }
}

// Print a process listing to console.  For debugging.
// Runs when user types ^P on console.
// No lock to avoid wedging a stuck machine further.
void procdump(void)
{
  static char *states[] = {
      [UNUSED] "unused",
      [SLEEPING] "sleep ",
      [RUNNABLE] "runble",
      [RUNNING] "run   ",
      [ZOMBIE] "zombie"};
  struct proc *p;
  char *state;

  printf("\n");
  for (p = proc; p < &proc[NPROC]; p++)
  {
    if (p->state == UNUSED)
      continue;
    if (p->state >= 0 && p->state < NELEM(states) && states[p->state])
      state = states[p->state];
    else
      state = "???";
    printf("%d %s %s", p->pid, state, p->name);
    printf("\n");
  }
}
