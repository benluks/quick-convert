Donors are components that are dedicated to a specific pipeline. It's code that's adapted with minimal (or possibly no) changes. It's not meant to fit the modularity of the rest of codebase, but rather be a quick way of integrating a system or component into this codebase without heavy refactoring.

**A unique donor need not and should not be useful for any module, other than that for which it was included!**

Ideally, a donor is a temporary addition. Once the pipeline in question works, we would integrate its components modularly in the code.

## How Donors Work

a donor exists under `quick_convert/components/donors/<donor_name>`. It should include:
* A README that instructs the user on how to setup the code (including necessary installations, downloads, or script runs)
* * **FUTURE**: Maybe these shuould have a setup.sh script that automatically runs the the setup. I left it out for now because I find that these scripts never work as planned.
* A `.gitignore`. This may be important, as donors will often require installing checkpoints, pre-computed features, etc. which should absolutely NOT be committed, and including these in the base `.gitignore` might betray the modularity that this code is built for.

## Existing Donors

* ### `emotion-compensation`

The only existing one (as of yet) is `emotion-compensation`. It exists solely for the purposes of instantiating the [`emotion_compensation`](https://github.com/benluks/quick-convert/blob/main/quick_convert/pipelines/anonymization/emotion_compensation.py) anonymization system.