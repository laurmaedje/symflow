# Symflow
**Data flow analysis for x86-64 ELF binaries based on symbolic execution.** 🔎

Symflow tracks the flow of values throughout the execution of a program and builds a _data dependency graph_, a structure describing how values flow through registers and memory locations within the program, enabling further analysis in need of such information. Along normal location nodes it also contains nodes for input and output, which represent bytes read from the standard input or written to the standard output, and constant nodes for constant values. Sometimes a value read from the input is only written to the output in specific circumstances. To account for such flows, the graph annotates edges with conditions in which the data flows.

This work is a _proof of concept_ and works only on a very small subset of _x86-64_ binaries.

## Building
This crate depends on the Z3 theorem solver and needs its library installed.
```bash
# Install the Z3 development libraries with apt
# or the package manager of your choice.
apt install libz3-dev
cargo build

# To run the tests and see the graph outputs, you will need
# to compile the examples with make and install graphviz.
cd test && make
apt install graphviz
cargo test
```

## Example
The following code performs some pointer arithmetic based on values read from the standard input. In this example, the buffers are arranged in such a way that the secret value read from standard input is written to the output if `x = 64 + y` holds true. This can also be seen in the data dependency graph shown below: The secret byte corresponds to the third byte read from standard input, namely `stdin2` (starts at zero). The (only) value that is written to the output corresponds to `stdout0`. A chain of arrows through a lot of registers and memory locations exists from `stdin2` to `stdin0` in the graph (at the right end), with one arrow holding exactly the condition discussed above.

#### Code
```c
void main() {
    char buf[1024];

    unsigned char x = read_one_byte();
    unsigned char y = read_one_byte();
    char secret = read_one_byte();

    char* a = buf + x;
    char* b = buf + 64 + y;

    a[x] = secret;
    char s = b[x];

    write_one_byte(s);
}
```

#### Data Dependency Graph
![Data Dependency Graph](https://github.com/laurmaedje/symflow/blob/master/Graph.png)
