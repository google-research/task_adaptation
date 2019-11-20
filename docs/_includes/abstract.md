The Visual Task Adaptation Benchmark (VTAB) is a diverse and challenging suite of tasks,
designed to evaluate general visual representations.

VTAB defines a good general visual representation as one that yields good performance on *unseen* tasks,
when trained on limited task-specific data.
VTAB places no restrictions on how the representations are used, for example, frozen feature extraction,
fine-tuning, and other forms of transfer to the evaluation tasks are permitted.
Similarly, representations may be pre-trained on any data, VTAB permits supervised, unsupervised, or other pre-training strategy.
There is one constraint: *the evaluation datasets must not be used during pre-training*.
This constraint is designed to mitigate overfitting to the evaluation tasks.

The benchmark consists of 19 tasks, drawn from a variety of domains, and with various semantics.
All tasks are framed as classification problems to facilitate a consistent API for pre-trained models.
Algorithms should not contain any task-dependent logic, for example,
the same hyperparameter sweep should be used for all tasks.
VTAB may also be used to evaluate techniques, other than representation learning,
that improve performance across a variety of tasks: such as architectures, pre-processing functions, or optimizers.

The page tracks the performance of algorithms published using VTAB.
To highlight a result to the VTAB admins, contact vtab@google.com.

VTAB provides:
* [Code](https://github.com/google-research/task_adaptation) to run the benchmark.
* [A repository](https://tfhub.dev/vtab) of pre-trained models, evaluated on the benchmark.
* A public leaderboard to track progress.
* *Coming Soon*: A mechanism to submit TF Hub modules for automatic evaluation.


[Paper](https://arxiv.org/abs/1910.04867)<br/>
[GitHub](https://github.com/google-research/task_adaptation)<br/>
[Models on TF Hub](https://tfhub.dev/vtab)
