# First, show how many nodes we have. 2 worker & 1 master
kgn 
# check current gpu usage
kubectl-inspect-gpushare 
nvidia-smi

# show run.sh to see entire workflow
# we will run 4 batch of jobs. run2 shows online job execution, while run3 & 4 shows jobs within mem limit will be scheduled first
# run 1 -> sleep 100s -> run2 -> complete 1 run2 job -> run 3&4 
vim run.sh



# run script
bash run.sh

# after each run, we could check profile results within csv format
# wait for the first run to finish...
# first run includes 35 jobs, in order to simulate gpu is fully utilized... (each profiled img job is 1747 Mb - capacity - 24564MB/1747MB =  14 jobs per GPU)

# run 1 profile completed
# run 2 profiling...Completed!
# run 3,4 profiling... Completed
# submitted run1 to cluster
# run1,2 submitted
# next job - run2-job1
# Once run2-job-1 running, submits run3 & run4
# NEXT - run4-job1


# once jobs are submitted, open another terminal to check reosurce usage gpu usage should be near fully-utilized with shorter job executed first
# lets check current schedule order...
kgpall
kubectl-inspect-gpushare


# GLOBAL execution sequence 
# RUN1 

# submit run3 & run4 together. We want to see if job that fits in GPU memory got scheduled first
kubectl apply -f ../k8sJobs/job_run3.yaml
kubectl apply -f ../k8sJobs/job_run4.yaml

# check how many pods on  a worker
kgpall --field-selector spec.nodeName=node5
kgpall --field-selector spec.nodeName=node6

# check current scheduled order
sed -n '$=' ~/kubernetes-scheduling/deployment/logs/sched.log
bash ~/kubernetes-scheduling/deployment/logs_defaultsched.sh
cat -n ~/kubernetes-scheduling/deployment/logs/sched.log | grep Successfully

# manually submit next 2 runs to simulate online arrival

kubectl apply -f ../k8sJobs/job_run4.yaml
# lets check out scheduler logs...

# new shortest job dont get preempted and will wait for current job to finish
kgpall
kubectl-inspect-gpushare 
nvidia-smi 

# The log accumulates, so we could find where current log ends
sed -n '$=' ~/kubernetes-scheduling/deployment/logs/sched.log

# check scheduler logs 
bash ~/kubernetes-scheduling/deployment/logs_defaultsched.sh

# trace the following log starting from the end of line we found
cat -n ~/kubernetes-scheduling/deployment/logs/sched.log | grep Successfully