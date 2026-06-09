#ifndef PLUGIN_H
#define PLUGIN_H

typedef char* (*plugin_fn)(int argc, char** argv);

void plugin_register(const char* name, plugin_fn fn);
plugin_fn plugin_lookup(const char* name);


// Declare plugin functions
char* plugin_ddg(int argc, char** argv);
char* plugin_summarize_file(int argc, char** argv);
char* plugin_websearch(int argc, char** argv);
char* plugin_summarize_file_html(int argc, char** argv);


#endif
