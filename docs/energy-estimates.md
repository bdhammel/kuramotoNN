# Energy estimates: numbers, sources and assumptions

This is the companion to [`pymoto.energy`](../pymoto/src/pymoto/energy.py). It
records every formula and constant the estimate uses, where each number comes
from, how far it has been checked, what is assumed and what is left out. Read it
before quoting any number the tool prints.

The short version:
- **The digital side rests on published circuit figures.**
- **The physical-oscillator side rests on placeholders.** They are
  order-of-magnitude guesses for CMOS ring oscillators, not measurements. The
  two least certain of them (coupler power and wire energy) make up 88% of the
  MNIST physical-core estimate.
- **The headline conclusions hold up against those placeholders.** They survive
  errors of 25× or more (see [Sensitivity](#sensitivity)).
- **The exact ratios do not.**

Regenerate any number below with:

```sh
python -m pymoto.energy kuramoto_mnist --energy-baseline 784,256,10
python experiments/mnist/eval.py --checkpoint runs/final --energy-only
```

## 1. What is being estimated

Energy per sample (per inference). It is always counted the same way: the number
of events of each kind, times the energy per event, summed.

The model is costed in two ways:

| accounting | events | used for |
|---|---|---|
| **digital** | multiply-accumulates (MACs); weight reads from on-chip SRAM or off-chip DRAM/HBM | the MLP baselines; the Kuramoto network *simulated* with Euler steps |
| **physical** | DAC writes, oscillator cycles, coupler power × time, coupling-wire charging, ADC conversions; plus the frozen `W` and `H`, which stay digital | the Kuramoto network *run as real oscillators* |

The general form comes from Murmann's energy-per-inference decomposition
([§8](#8-context-from-the-literature)):

```
E/inference ≈ [ E/op + (memory accesses / op) · (E / memory access) ] · ops/inference
```

The physical accounting replaces "ops" with physical events, and sets the memory
term to zero for K, because K lives in the coupling hardware.

## 2. Digital accounting

### 2.1 Workload counts

Per sample, batch 1. These counts are **exact for the code as written**.
`tests/test_energy.py` checks the MAC totals against `torch.utils.flop_counter`
on a real forward pass.

| stage | MACs | distinct weights | on-chip weight reads |
|---|---|---|---|
| drive `z = x @ W.T` | `in_dim · n` | `in_dim · n` | `in_dim · n` |
| coupling, per Euler step | `2 · n²` | `n²` (K, once) | `n²` |
| coupling, whole rollout | `2 · n² · num_steps` | `n²` | `n² · num_steps` |
| head `features @ H.T` | `2n · C` | `2n · C` | `2n · C` |
| MLP layer `a → b` | `a · b` | `a · b + b` (bias) | `a · b + b` |

The coupling rows assume the following, per `layers/coupling.py`:
- Each Euler step runs two dense `(1, n) @ (n, n)` products, `sin θ @ Kᵀ` and
  `cos θ @ Kᵀ`.
- **One read of K serves both products.** This assumes a fused kernel; an unfused
  one would read K twice per step.
- K's diagonal is included, because the matmul is dense.

For the registered models:

| model | MACs | weights | of which K | notes |
|---|---|---|---|---|
| `kuramoto_mnist` (n=256, 10 steps) | 1,516,544 | 271,360 | 65,536 | 86% of MACs are coupling |
| `kuramoto_mnist`, 50 steps (`runs/final.pt`) | 6,759,424 | 271,360 | 65,536 | |
| `kuramoto_cartpole` (n=64, 10 steps) | 82,432 | 4,608 | 4,096 | |
| MLP 784-256-10 | 203,264 | 203,530 | — | reference only; not in the repo |
| MLP 4-128-2 (CartPole policy) | 768 | 898 | — | |

### 2.2 Memory model

- **`sram`**: the whole model sits in about 1 MB of on-chip SRAM. Every weight
  read costs one SRAM access. At int8, MNIST's 271 KB fits; at fp32 (1.08 MB) it
  is borderline.
- **`dram`**: each distinct weight is fetched once from DRAM/HBM. Any reuse
  beyond that first read (K on later Euler steps) comes from SRAM.
- **Batch size `B`** (`--energy-batch-size`): weight traffic is divided by `B`,
  because one read serves the whole batch. Compute is not divided. The default
  is 1: the latency-bound or edge case, and also how a physical oscillator
  network runs, one sample at a time.

```
E_compute = MACs · (E_mul + E_add)
E_memory  = [sram] weight_reads · E_sram · bytes_per_weight / B
            [dram] (weights · E_dram + (weight_reads − weights) · E_sram) · bytes_per_weight / B
```

### 2.3 Digital energy constants

| preset | E_mul | E_add | bytes/weight | SRAM, per byte | DRAM/HBM, per byte |
|---|---|---|---|---|---|
| `7nm-int8` (default) | 0.07 pJ | 0 | 1 | 2 pJ | 39 pJ |
| `45nm-int8` | 0.2 pJ | 0.1 pJ | 1 | 12.5 pJ | 162.5 pJ |
| `45nm-fp32` | 3.7 pJ | 0.9 pJ | 4 | 12.5 pJ | 162.5 pJ |

Where each number comes from, and how far it has been checked:

**`7nm-int8`: Google, ISCA 2021** (doi:10.1109/ISCA52012.2021.00010).
- **Where the numbers were taken from:** the unconv.ai post "How to improve AI
  energy efficiency by 1000x" (7 May 2026), read in full on 2026-09-23. It gives
  "0.07 pJ" per 8-bit integer multiply, "2 pJ (per 8 bits)" for SRAM reads, and
  "39 pJ (per 8 bits)" for HBM reads.
- **The SRAM figure is optimistic.** The post's footnote says it "assumes the use
  of 1 MB (not larger) SRAMs and that you can use the result locally".
- **Adds are set to 0** because the source quotes multiplies only. Integer adds
  are roughly 10× cheaper than multiplies, so this undercounts compute slightly.
  At these sizes compute is only about 6% of the digital total (5.8% for MNIST,
  6.5% for CartPole), so it does not matter.
- **The primary ISCA paper has not been checked here.**

**`45nm-*`: Horowitz, "Computing's energy problem (and what we can do about it)",
ISSCC 2014** (doi:10.1109/ISSCC.2014.6757323). 45 nm at 0.9 V.
- **Arithmetic:** 8-bit multiply 0.2 pJ; 32-bit add 0.1 pJ (used as the int8
  accumulator, so a MAC is 0.3 pJ); fp32 multiply 3.7 pJ; fp32 add 0.9 pJ.
- **Memory:** per 64-bit access, a 1 MB SRAM costs 100 pJ, giving 12.5 pJ/byte.
  DRAM costs 1.3–2.6 nJ; the low end is used, giving 162.5 pJ/byte.
- **These are the widely reproduced values from the paper's table.** They were
  written from that table, not re-checked against the paper in this work.
- **Newer nodes lower compute energy much more than DRAM energy.** Treat the
  45 nm presets as an upper bound on compute.

**What's not in the digital accounting:**
- **Activation traffic**, which is O(n) per sample against O(n²) weights.
- **Elementwise work.**
- **The sin/cos evaluations:** 2n per step plus 2n in the readout, which is 5,632
  for MNIST against 1.5M MACs.

## 3. Physical accounting

### 3.1 Equations

The central conversion is time. The model integrates for `T` units of model time,
and one unit is `cycles_per_unit_time` carrier cycles:

```
cycles = T · cycles_per_unit_time           t_run = cycles / carrier_hz
N_c    = #{ i ≠ j : |K_eff[i,j]| > threshold }    (couplers; the diagonal is never built)

E_set_frequencies = n · E_dac                                   one write per oscillator to set g·z_i
E_oscillators     = n · E_cycle · cycles
E_couplers        = N_c · P_coupler · t_run
E_wires           = N_c · pitch · c_wire · V² · cycles
E_readout         = n · conversions_per_oscillator · E_adc
E_core            = sum of the five above

E_drive, E_head   = digital stages (§2), weights read from SRAM, DRAM, or regenerated
                    from the seed:  MACs · (E_mul + E_add) + weights · E_rng
E_total           = E_core + E_drive + E_head
```

`num_steps` does not appear in these equations. The hardware integrates the
continuous flow, so only `T` matters. `tests/test_energy.py` checks this.

### 3.2 Hardware constants: placeholders

**None of these values are measurements.** They are order-of-magnitude guesses
for a CMOS ring-oscillator implementation, chosen so the terms are the right
size relative to one another. Override any of them with `--hw field=value`.

| field | default | where it comes from | confidence |
|---|---|---|---|
| `carrier_hz` | 1 GHz | typical CMOS ring-oscillator frequency | assumption |
| `cycles_per_unit_time` | 100 | see below | **assumption; sets every time-proportional term** |
| `energy_per_cycle_j` | 50 fJ | 5-stage ring × ~5 fF per stage × (1 V)² ≈ 25 fJ; a range of 10–100 fJ assumed | back-of-envelope |
| `coupler_power_w` | 1 µW | an active coupler biased at ~1 µA from ~1 V | **pure placeholder, no source** |
| `wire_cap_f_per_um` | 0.2 fF/µm | common rule of thumb for on-chip interconnect | rule of thumb |
| `coupler_pitch_um` | 10 µm | assumed crossbar cell size | **assumption** |
| `supply_v` | 0.8 V | assumed | assumption |
| `dac_j_per_conversion` | 1 pJ | assumed, similar to the ADC | placeholder |
| `adc_j_per_conversion` | 1 pJ | an ~8-bit converter: a few fJ per conversion step × 2⁸ steps. Murmann's ADC Performance Survey is the place to check | placeholder |
| `conversions_per_oscillator` | 2 | the features `[sin θ, cos θ]` are exactly an I/Q demodulation against a reference | structural |
| `rng_j_per_weight` | 0 | not modelled; an LFSR emitting ±1 entries would cost well under a MAC | **not modelled** |

**`cycles_per_unit_time`**:
- **Why it's much greater than 1:** real oscillators follow Kuramoto phase
  dynamics only under weak coupling, where the coupling rate is far below the
  oscillator's damping rate ω₀/2Q. So one unit of model time spans many carrier
  cycles.
- **Where 100 comes from:** it's a guess within "tens to hundreds".
- **What would set it in a real design:** its phase noise and coupling strength.

**Wire model:**
- **Layout:** each coupler adds one pitch of wire to the line its source
  oscillator drives, as in a compact crossbar.
- **Charging:** that line charges and discharges rail-to-rail once per cycle,
  costing `C·V²`.
- **Why it's counted:** dense all-to-all coupling means every oscillator's
  signal is sent across the chip continuously. That is data movement in another
  form, which is why the term exists.
- **Scaling:** a sparse K needs fewer couplers and shorter wires. Setting
  `coupler_threshold` makes the estimate reflect that.

### 3.3 Assumptions built into the physical model

1. **The hardware reproduces the simulated accuracy.** The estimate says nothing
   about whether it does. The solver-transfer control (Euler vs RK4 at 10× the
   resolution) is the first check, because hardware integrates the exact flow,
   not the Euler grid.
2. **Every coupler draws the same power.** An asymmetric K needs active,
   non-reciprocal couplers. A symmetric K could use passive resistive or
   capacitive ones. The model does not distinguish the two cases.
   - Setting `coupler_power_w=0` approximates passive couplers: the MNIST core
     falls from 16.9 to 10.4 nJ.
   - The `k_asymmetry` diagnostic is therefore also an energy diagnostic.
3. **`W` and `H` are computed digitally,** at the same precision as the digital
   preset.
4. **The whole K fits in the fabric.** If `n` exceeds what the chip can hold,
   the couplers would have to be reprogrammed during inference, which the model
   does not include.
5. **Phases are read once, at `t = T`.**

### 3.4 What's not in the physical model

- **Writing K into the coupling fabric.** It happens once per model and is
  amortized over every inference.
- **Idle and static leakage,** bias generation, and distribution of the clock
  and reference.
- **The reference oscillator needed for I/Q demodulation.**
- **The mean-phase subtraction in the readout.** The code subtracts the
  arithmetic mean of unwrapped phases (`layers/readout.py`). Hardware sees only
  θ mod 2π, so it would have to use the circular mean, arg Σ e^{iθ}. Accuracy
  with that change has not been tested.
- **Precision and noise.** Nothing ties the energy to the number of bits the
  network needs (see [§8](#8-context-from-the-literature)):
  - The kT/C law says each extra bit of precision costs 4× the energy.
  - For oscillators, a rough thermal-noise floor is
    `E ≳ n · (Γ·t_run)² · kT / (2σ_θ²)` per inference. Here Γ is the damping
    rate and σ_θ is the phase error the network can tolerate.
  - The floor is an order-of-magnitude derivation that is not in the code. A
    phase-noise sweep would pin σ_θ down.
- **Frequency mismatch.** The model assumes every oscillator has zero natural
  frequency apart from its drive, which in hardware means trimming each one.
  Trimming precision and its cost are not modelled.
- **Amplitude dynamics and non-sinusoidal coupling** in real oscillators.
- **Getting the input `x` onto the chip.**

## 4. Reference outputs

These use the default presets (`7nm-int8`, the placeholder hardware), batch 1
and `T = 1`, giving 100 cycles = 100 ns. All values are nJ per sample.

| | MNIST (n=256, 10 steps) | CartPole (n=64, 10 steps) |
|---|---|---|
| digital simulation, weights in SRAM | 1,829 | 88.7 |
| digital simulation, weights from DRAM/HBM | 11,869 | 259 |
| physical core | 16.9 | 1.43 |
| — DAC writes / oscillators / couplers / wires / ADC readout | 0.26 / 1.28 / 6.53 / 8.36 / 0.51 | 0.06 / 0.32 / 0.40 / 0.52 / 0.13 |
| physical total, `W`/`H` in SRAM | 443 | 2.49 |
| physical total, `W`/`H` from DRAM/HBM | 8,058 | 21.4 |
| physical total, `W`/`H` from seed | 31.3 | 1.47 |
| MLP baseline, SRAM (784-256-10 / 4-128-2) | 421 | 1.85 |
| `sim_over_core` | 108× | 62× |
| `drive_head_share` (SRAM) | 96% | 43% |
| MLP / physical, `W`/`H` in SRAM | 0.95× | 0.74× |
| MLP / physical, `W`/`H` from seed | 13.4× | 1.26× |

- **Your trained `runs/final.pt`** was trained with 50 steps. Its simulation
  costs 7,438 nJ (SRAM), 439× the core. Every physical number is unchanged,
  since the physical model has no steps.
- **Most of the simulation's cost is memory.** Of the MNIST digital figure
  (SRAM), 94% is weight reads (1,722 nJ) and 6% is arithmetic (106 nJ).

## 5. Sensitivity

**The physical core in closed form.** It is linear in `cycles_per_unit_time`,
with `T = 1` and everything else at its default:

```
MNIST:    E_core ≈ 0.77 nJ + 0.162 nJ · cycles_per_unit_time
CartPole: E_core ≈ 0.19 nJ + 0.012 nJ · cycles_per_unit_time
```

**For MNIST, wires (49%) and couplers (39%) make up 88% of the core.** Those are
the two least-supported numbers. The oscillators are 8%, the ADC readout 3% and
the DAC writes 2%.

**How far the conclusions survive errors in the placeholders:**

| conclusion | holds unless the core estimate is too low by more than |
|---|---|
| MNIST: running physically is cheaper than simulating (`sim_over_core > 1`) | 108× (439× for the 50-step checkpoint) |
| MNIST: the digital `W`, `H` dominate the physical total (`drive_head_share > 50%`) | 25× |
| CartPole: running physically is cheaper than simulating | 62× |
| CartPole: `W`, `H` dominate | does not hold now (43%); the core already dominates |

For each conclusion:
- **MNIST's Amdahl conclusion is robust.** The `W` projection dominates the
  energy unless it stops being read from memory.
- **MNIST's "cheaper than simulating" conclusion is robust.**
- **Any ratio against the MLP is not robust.** CartPole's 0.74× and 1.26× could
  flip with modest changes to the placeholders.
- **MNIST's 13.4× figure (`W`/`H` from seed) also depends on the placeholders.**
  It depends on the core, because once `W` stops being read from memory, the
  core is half of the total.
- **Every ratio against the MLP assumes the two models reach the same
  accuracy.** Nothing checks that.

## 6. Where the numbers could be improved, in order of payoff

1. **Coupler power and the wire model:** 88% of the MNIST core. Replace them with
   per-oscillator and per-coupler power reported for a real coupled-oscillator
   chip, for example Moy et al. 2022.
2. **`cycles_per_unit_time`:** it scales every time-proportional term. Estimate it
   from a target phase-noise budget and the locking range.
3. **Precision:** a phase-noise sweep and quantization of K and the readout in
   simulation would give the bits the network actually needs. Then the kT/C and
   ADC-survey scaling would set the DAC, ADC and oscillator energies instead of
   guesses.
4. **Primary sources for the 7 nm preset:** check the ISCA 2021 paper directly.
5. **An MLP baseline at matched accuracy** for MNIST, so the ratios mean
   something.

## 7. Where each piece lives in the code

| item | location |
|---|---|
| presets | `DigitalCosts`, `GOOGLE_7NM_INT8`, `HOROWITZ_45NM_INT8`, `HOROWITZ_45NM_FP32` in `pymoto/src/pymoto/energy.py` |
| placeholders | `OscillatorHardware` in the same file |
| counts | `kuramoto_workload`, `linear_workload`, `mlp_workload` |
| equations | `digital_energy`, `physical_energy`, `count_couplers` |
| report keys | `energy_report` (all `energy/*`, values in nJ) |
| tests | `pymoto/tests/test_energy.py` |

## 8. Context from the literature

These points are not coded as constants. They explain how to read the output.
They come from B. Murmann, "Analog is dead, long live analog!" (unconv.ai, 30 Apr
2026) and the 1000x post above, both read in full on 2026-09-23:

- **The precision law.** Thermal noise scales as √(kT/C), so "achieving an extra
  bit of precision quadruples the required capacitance, hence quadrupling the
  energy." Analog wins at 1–4 bits and loses above about 8.
- **Analog arithmetic alone is not a large win.** Digital 4-bit MACs in in-memory
  compute reach ">100 TOPS/W (10 fJ/op)". The best analog reaches "several hundred
  TOPS/W". The gain has to come from not reading memory.
- **Memory is the problem.** In the 1000x post's 100B-parameter example, going
  from arithmetic only, to SRAM reads, to HBM reads multiplies energy per token
  by more than 500×. The post says: "If we kept the energy cost of MAC arithmetic
  operations the same and eliminated all energy costs from memory, we would
  likely immediately get a ~1000x energy advantage."
- **Moving data across a chip costs about 2.4 pJ/byte.** This is what motivates
  the wire term, although the term is modelled from wire capacitance rather than
  this figure.
- **Amdahl's law.** Any part left unoptimized caps the total gain. Here, that
  part is the digital `W`.
- **The "wind tunnel" test.** "If a circuit requires less energy to simulate with
  a current digital processor than it does to run physically, then it is not a
  promising candidate." That is what `sim_over_core` measures.
- **Iso-quality.** The fair baseline is the best conventional model at the same
  accuracy, not the network's own simulation.
- **Noise costs a larger model.** "If the design comes with significant analog
  noise, a larger model may be required for iso-accuracy." For this network, a
  larger `n` grows the couplers and wires as n².

## References

- M. Horowitz, "Computing's energy problem (and what we can do about it)," ISSCC 2014. doi:10.1109/ISSCC.2014.6757323
- Google, ISCA 2021 (TPUv4i). doi:10.1109/ISCA52012.2021.00010. The 7 nm figures here are taken from the unconv.ai post, not from the paper.
- Unconventional AI, "How to improve AI energy efficiency by 1000x," 7 May 2026. https://unconv.ai/blog/how-to-improve-ai-energy-efficiency-by-1000x/
- B. Murmann, "Analog is dead, long live analog!," Unconventional AI, 30 Apr 2026. https://unconv.ai/blog/analog-is-dead-longlive-analog/
- B. Murmann, "Mixed-Signal Computing for Deep Neural Network Inference," IEEE TVLSI, Jan. 2021. This is the analog-vs-digital dot-product energy model behind the 4^B law.
- B. Murmann, ADC Performance Survey, for converter energy against resolution.
- G. Csaba and W. Porod, "Coupled oscillators for computing: A review and perspective," Applied Physics Reviews 7, 011302 (2020). doi:10.1063/1.5120412
- W. Moy et al., "A 1,968-node coupled ring oscillator circuit for combinatorial optimization problem solving," Nature Electronics 5, 310–317 (2022).
