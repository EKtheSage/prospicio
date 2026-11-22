### Excel

- A finished asynchronous call no longer replies to Excel after the add-in closes (a shutdown flag set when Excel unloads it), so a run that finishes while Excel quits no longer calls into it. The next call to any function turns replies back on, so a quit cancelled at the save prompt leaves the add-in working (#955, #1036).
