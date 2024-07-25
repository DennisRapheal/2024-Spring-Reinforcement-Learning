
# Conservative Q-Learning for Offline Reinforcement Learning 
 
## Brief Summary
Conservative Q-Learning, CQL, aims to solve the overestimaion of values in offline RL algortihms by learning a conservative Q-function such that the expected value of a policy under this Q-function lower-bounds its true value. 

## Preliminaries
### Notations 
* $T(s'|s,a)$ : The probability of $s\rightarrow a \rightarrow s'$
* $\mathcal{D}$ : the dataset of states sampled from $d^{\pi_{\beta}}(s)\pi_{\beta}$
* $r(s,a)$ : reward, $\gamma$ :discount factor
* ${\pi}_{\beta}(a|s)$ : behavior policy
* $\hat{\pi}_{\beta}(a|s)$ := $\frac{\sum_{s, a \in \mathcal{D}}1[s=s, a=a]}{\sum_{s, a \in \mathcal{D}}1[s=s]}$ : the empirical behavior policy
* $d^{\pi_{\beta}}(s)$ : the discount marginal state-distribution of $\pi_{\beta}(a|s)$
* $\mu (s,a)$ : a distribution of state-action pairs


### Assumptions
* Assume the reward is bounded : $|r(s,a)| \leq R_\text{max}$


### Original Q-learning 
* Bellman operator

    $\mathcal{B}^{\pi}Q = r + \gamma P^{\pi}Q$ ,   $\ \ \ P^{\pi}Q(s,a) = \mathbb{E}_{s'\sim T(s'|s, a), a'\sim \pi(a'|s')}\left[ Q(s', a') \right]$
    
* train Q-function by iteratively applying **Bellman optimality operator**: 

    $\mathcal{B}^*Q(s,a) = r(s,a) + \gamma\mathbb{E}_{s'\sim P(s'|s,a)}[\max_{a'}Q(s',a')]$
    
* The empirical Bellman operator $\hat{\mathcal{B}}^{\pi}$ only contains a single sample in $\mathcal{D}$. 

    $\hat{\mathcal{B}}^{\pi} \hat{Q}^k(s,a) = r(s,a) + \gamma \mathbb{E}_{a'\sim\hat{\pi}(a'|s')}[\hat Q^k(s',a')]$

* Try to constrain $a'$ to stay close to behavior policy in Q-learning. In this case, it is able to avoid querying $Q$ on out-of-distribution(OOD) actions. Set a original Q-learning policy evaluation :

    $\begin{align*} \hat{Q}^{k+1} &\gets \arg\min_Q \mathbb{E}_{s,a,s'\sim D} \left[\left((r(s,a) + \gamma\mathbb{E}{a'\sim\hat{\pi}^k(a'|s')}[\hat{Q}^k(s',a')]) - Q(s,a)\right)^2\right]  \end{align*}$

* Original Q-learning policy improvement

    $\begin{align*} \hat{\pi}^{k+1} &\gets \arg\max_\pi \mathbb{E}_{s\sim D,a\sim\pi^k(a|s)}\left[\hat{Q}^{k+1}(s,a)\right] \end{align*}$

### Problems in Offline RL Algorithms
In practice, offline RL has encountered major challenge in offline RL is the distribution shift between the dataset and the learned policy, which can lead to overestimation of values for out-of-distribution actions.


## The Framework of Conservative Q-Learning
### Conservative Off-Policy Evaluation
* To be conservative, try to add penalty into the policy evaluation algorithm. The paper chose to restrict $\mu$, by letting $\mu(s,a) = d^{\pi_{\beta}}(s)\pi_{\beta}$. Simply view the blue part and the coefficient $\frac{1}{2}$ as a penalty. 

    $\hat{Q}^{k+1} \leftarrow {\arg\min}_{Q}\ \ \color{blue}{\alpha\mathbb{E}_{s\sim D, a\sim\mu(a|s)}[Q(s,a)]} \ + \frac{1}{2}\mathbb{E}_{s,a,s'\sim D}\left[\left(Q(s,a) -\hat{B}^{\pi}\hat{Q}^k(s,a)\right)^2\right] ... (1)$
* It has been proved in theorem3.1 that the equation(1) lower-bounds the true Q-function $Q^{\pi}$. (Find theorem3.1 in **Supporting Lemma and Theoretical Analysis**)

* The bound is tightenable if we only required the expected value of $\hat{Q}^{\pi}$ , under policy $\pi(a|s)$, and lower-bound $V^{\pi}$, we can improve the bound by introducing an additional Q-value maximization term. Revise the fisrt (1) equation as: 

    $\hat{Q}^{k+1} \leftarrow {\arg\min}_{Q}\ \ \alpha \left( \mathbb{E}_{s\sim D, a\sim\mu(a|s)}[Q(s,a)] \color{red}{ -\mathbb{E}_{s\sim D, a\sim\hat{\pi}_{\beta}(a|s)}[Q(s,a)]} \right)  \ + \frac{1}{2}\mathbb{E}_{s,a,s'\sim D}\left[\left(Q(s,a) -\hat{B}^{\pi}\hat{Q}^k(s,a)\right)^2\right]...(2)$

* It has been proved in theorem3.2 that the equation(2) lower-bounds the expected value under the policy $\pi$, when $\mu = \pi$.(Find theorem3.2 in **Supporting Lemma and Theoretical Analysis**)
### Conservative Q-Learning for Offline RL
* With taking $\mu(s,a) = d^{\pi_{\beta}}(s)\pi_{\beta}$, since the policy $\hat{\pi}^k$ is derived from Q-function, we could instead choose $\mu(s,a)$ to approximate the policy that would maximize the current Q-function iterate. 
* Denote the $\text{CQL}(\mathcal R)$ with a chosen regulizer $\mathcal{R}(\mu)$ : 

    $\min_Q{\color{red}{\max}}_{\color{red}{\mu}} \ \alpha \left( \mathbb{E}_{s \sim \mathcal{D}, a \sim \mu(a|s)} \left[ Q(s, a) \right] - \mathbb{E}_{s \sim \mathcal{D}, a \sim \hat{\pi}_\beta(a|s)} \left[ Q(s, a) \right] \right) + \frac{1}{2} \mathbb{E}_{s, a, s' \sim \mathcal{D}} \left[ \left( Q(s, a) - \hat{\mathcal{B}}^{\pi_k} \hat{Q}^k(s, a) \right)^2 \right] + \color{red}{\mathcal{R} (\mu)}\ \  \left( \text{CQL}(\mathcal{R}) \right)...(3)$

* "conservative" indicates the sense that each successive policy iterate is optimized against a lower bound on its value. $\text{CQL}$ is say to be conservative by showing that $\text{CQL}(\mathcal H)$ learns Q-value estimates that lower-bound the actual Q-function. It has been proved in theorem3.3 that $\text{CQL}$ learns lower-bounded Q-values with large enough $\alpha$, meaning that the final policy attains the estimated value. 

* The paper also shows that $\text{CQL}$ optimizes a well-defined, penalized empirical RL objective, and performs high-confidence safe policy improvement over the behavior policy. 
### Regulizer and Variant of CQL
* The paper provides some variant of CQL based on different requlizer. One of them is$\text{CQL}(\mathcal{H})$. (The disscussion of CQL variants is in Appendices P.14)
* Choose ${\mathcal{R} (\mu)}$ to be the KL-divergence against a prior distribution $\rho (a|s)$, i.e. $\mathcal{R}(\mu) = -D_{KL}(\mu, \rho)$ . $\text{CQL}(\mathcal H) :=$
    
    $\min_Q \ \alpha \mathbb{E}_{s \sim \mathcal{D}} \left[ \log \sum_a \exp(Q(s, a)) - \mathbb{E}_{a \sim \hat{\pi}_\beta(a|s)} \left[ Q(s, a) \right] \right] + \frac{1}{2} \mathbb{E}_{s, a, s' \sim \mathcal{D}} \left[ \left( Q - \hat{\mathcal{B}}^{\pi_k} \hat{Q}^k \right)^2 \right]$

## Algorithm Implementation
### Pseudocode

![IMG_8126732FCFBC-1](https://hackmd.io/_uploads/SJgeXl870.jpg)
* for both actor-critic and Q-learning variant, the implemenation basically swap the Bellman error with $\text{CQL}(\mathcal{R})$. 

### Note
* The provided algorithm only reauires an addition of 20 lines of code on top of standard implementations of soft actor-critic(SAC) for continuous control experiments and on top of QR-DQN for the discrete control.

## Supporting Lemmas and Theoretical Analysis
The proofs of following theorems can be found in Appendices.  

### Theorem 3.1
For any $\mu(a|s)$ *with* $\operatorname{supp} \mu \subseteq \operatorname{supp} \hat{\pi}_\beta$, *with probability* $\geq 1-\delta$, $\hat{Q}^\pi$ *(the Q-function obtained by iterating Equation (1)) satisfies:*

$$
\forall s \in \mathcal{D}, a, \quad \hat{Q}^\pi(s,a) \leq Q^\pi(s,a) - \alpha \left[ (I - \gamma P^\pi)^{-1} \frac{\mu}{\hat{\pi}_\beta} \right](s,a) + \left[ (I - \gamma P^\pi)^{-1} \frac{C_{r,T,\delta}R_{\max}}{(1-\gamma)\sqrt{|\mathcal{D}|}} \right](s,a).
$$

Thus, if $\alpha$ is sufficiently large, then $\hat{Q}^\pi(s,a) \leq Q^\pi(s,a), \forall s \in \mathcal{D}, a$. When $\hat{\pi}_\beta = \pi$, any $\alpha > 0$ guarantees $\hat{Q}^\pi(s,a) \leq Q^\pi(s,a), \forall s \in \mathcal{D}, a \in \mathcal{A}$.

Showing that the equation(1) lower-bounds the true Q-function $Q^{\pi}(s,a)$.
### Theorem 3.2
The value of the policy under the Q-function from Equation (2), $\hat{V}^\pi(s) = \mathbb{E}_{\pi(a|s)}[\hat{Q}^\pi(s,a)]$, lower-bounds the true value of the policy obtained via exact policy evaluation, $V^\pi(s) = \mathbb{E}_{\pi(a|s)}[Q^\pi(s,a)]$, when $\mu = \pi$, according to:

$$
\forall s \in \mathcal{D}, \quad \hat{V}^\pi(s) \leq V^\pi(s) - \alpha \left[ (I - \gamma P^\pi)^{-1} \mathbb{E}_\pi \left[ \frac{\pi}{\hat{\pi}_\beta} - 1 \right] \right](s) + \left[ (I - \gamma P^\pi)^{-1} \frac{C_{r,T,\delta}R_{\max}}{(1-\gamma)\sqrt{|\mathcal{D}|}} \right](s).
$$

Thus, if $\alpha > \frac{C_{r,T}R_{\max}}{1-\gamma} \cdot \max_{s \in \mathcal{D}} \frac{1}{\sqrt{|\mathcal{D}(s)|}} \cdot \left[ \sum_a \pi(a|s) \left( \frac{\pi(a|s)}{\hat{\pi}_\beta(a|s)} - 1 \right) \right]^{-1}$, then $\hat{V}^\pi(s) \leq V^\pi(s), \forall s \in \mathcal{D}$, with probability $\geq 1 - \delta$. When $\hat{\pi}_\beta = \pi$, then any $\alpha > 0$ guarantees $\hat{V}^\pi(s) \leq V^\pi(s), \forall s \in \mathcal{D}$. Showing that the equation(2) lower-bounds the expected value $V^\pi(s)$
### Theorem 3.3

Let $\pi_{\hat{Q}_k}(a|s) \propto \exp(\hat{Q}^k(s,a))$ and assume that $D_{TV}(\hat{\pi}^{k+1}, \pi_{\hat{Q}_k}) \leq \varepsilon$ (i.e., $\hat{\pi}^{k+1}$ changes slowly w.r.t $\hat{Q}^k$). Then, the policy value under $\hat{Q}^k$, lower-bounds the actual policy value, $\hat{V}^{k+1}(s) \leq V^{k+1}(s) \forall s \in \mathcal{D}$ if

$$
\mathbb{E}_{\pi_{\hat{Q}_k}(a|s)} \left[ \frac{\pi_{\hat{Q}_k}(a|s)}{\hat{\pi}_\beta(a|s)} - 1 \right] \geq \max_{a \text{ s.t. } \hat{\pi}_\beta(a|s) > 0} \left( \frac{\pi_{\hat{Q}_k}(a|s)}{\hat{\pi}_\beta(a|s)} \right) \cdot \varepsilon.
$$
This shows CQL RL learns lower-bound Q-values with large enough $\alpha$, meaning that the final policy attains at least the estimated value.
### Theorem 3.4
At any iteration $k$, CQL expands the difference in expected Q-values under the behavior policy $\pi_\beta(a|s)$ and $\mu_k$, such that for large enough values of $\alpha_k$, we have that $\forall s \in \mathcal{D}$,

$$
\mathbb{E}_{\pi_\beta(a|s)}[\hat{Q}^k(s,a)] - \mathbb{E}_{\mu_k(a|s)}[\hat{Q}^k(s,a)] > \mathbb{E}_{\pi_\beta(a|s)}[Q^k(s,a)] - \mathbb{E}_{\mu_k(a|s)}[Q^k(s,a)].
$$
Showing the Q-function is "gap-expanding", which implied that CQL will only **over-estimate** the gap between in-distribution and out-of-distribution actions, preventing OOD actions.

## Experiment Result 
### The Performance of $\text{CQL}(\mathcal{H})$ on D4RL. 
![IMG_E8A1CBDA985B-1](https://hackmd.io/_uploads/ryZVnl8Q0.jpg)

* Adroit tasks
![IMG_4014D4236B1E-1](https://hackmd.io/_uploads/B1-qinv7C.jpg)

* Difference between policy values predicted by each algorithm and the true policy value. (the all negative results of CQL showed the lower-bound indeed exist.)
![IMG_91CF9C65E2E4-1](https://hackmd.io/_uploads/B1hoeTvmA.jpg)







## Discussions and Takeaways
### Takeaways
CQL addresses overestimation in offline RL by penalizing Q-values of unseen actions with conservatively optimizing the policy against a lower bound on its value. CQL also ehances the safety and robustness of policies in real-world deployments.

### Discussions
I thought that conservative Q-learning is kind of a trade-off between performance and the robotness of Q-learning. We might need to come out with some new tricks to accelerate the convergence process; in other words, enhancing the performance. 

Additionally, the paper mentioned some future works:
> "While we prove that CQL learns lower bounds on the Q-function in the tabular, linear, and a subset of non-linear function approximation cases, a rigorous theoretical analysis of CQL with deep neural nets, is left for future work. Additionally, offline RL methods are liable to suffer from overfitting in the same way as standard supervised methods, so another important challenge for future work is to devise simple and effective early stopping methods, analogous to validation error in supervised learning. "

## References
:page_facing_up: [Original Paper Link](https://arxiv.org/pdf/2006.04779) 
 :santa: [Stanord PPT](https://cs224r.stanford.edu/slides/cs224r_offline_rl_1.pdf)