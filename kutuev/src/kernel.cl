__kernel void printHello() {
    unsigned id1 = get_group_id(0);
    unsigned id2 = get_local_id(0);
    unsigned id3 = get_global_id(0);

    printf("I am from %d block, %d thread (global index: %d)\n", id1, id2, id3);
}

__kernel void addIdx(__global unsigned* a) {
    unsigned id = get_global_id(0);
    a[id] += id;
}