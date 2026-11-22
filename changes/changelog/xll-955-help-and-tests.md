### Excel

- A finished asynchronous call no longer replies to Excel after the add-in closes (a shutdown flag set when Excel unloads it), so quitting Excel during a long fit cannot make a worker call into it.
