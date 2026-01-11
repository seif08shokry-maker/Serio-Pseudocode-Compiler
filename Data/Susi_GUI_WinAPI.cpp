#include <windows.h>
#include <string>
#include <vector>

// Include your compiler's API header
#include "susi_compiler_api.h"


// Constants for our GUI elements
#define ID_EDITOR 101
#define ID_COMPILE_BUTTON 102


// Global handles to our GUI controls
HWND g_hEditor;
HWND g_hCompileButton;


// Function to convert a wide string to a standard string
std::string WideStringToString(const std::wstring& wstr) {
    if (wstr.empty()) return std::string();
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string str_to(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &str_to[0], size_needed, NULL, NULL);
    return str_to;
}


// Window procedure function to handle messages
LRESULT CALLBACK WndProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
    case WM_CREATE:
    {
        // Create the multiline text editor
        g_hEditor = CreateWindowExW(
            WS_EX_CLIENTEDGE,
            L"EDIT",
            L"Write your Serio pseudocode here...",
            WS_CHILD | WS_VISIBLE | WS_VSCROLL | ES_MULTILINE | ES_AUTOVSCROLL | ES_WANTRETURN,
            10, 10, 760, 500,
            hwnd,
            (HMENU)ID_EDITOR,
            GetModuleHandle(NULL),
            NULL
        );


        // Create the compile button
        g_hCompileButton = CreateWindowW(
            L"BUTTON",
            L"Compile & Run",
            WS_TABSTOP | WS_VISIBLE | WS_CHILD | BS_DEFPUSHBUTTON,
            10, 520, 760, 30,
            hwnd,
            (HMENU)ID_COMPILE_BUTTON,
            GetModuleHandle(NULL),
            NULL
        );
        break;
    }
    case WM_COMMAND:
    {
        // Handle button clicks
        if (LOWORD(wParam) == ID_COMPILE_BUTTON) {
            int textLength = GetWindowTextLengthW(g_hEditor); // Use the wide version of this function too
            std::vector<wchar_t> buffer(textLength + 1);
            GetWindowTextW(g_hEditor, buffer.data(), textLength + 1); // Use GetWindowTextW
           
            std::wstring wstr(buffer.data());
            std::string sourceCode = WideStringToString(wstr);


            // Call your compiler function
            compileSusi(sourceCode);


            // Show a success message
            MessageBoxW(hwnd, L"Compilation successful!", L"Success", MB_OK | MB_ICONINFORMATION);
        }
        break;
    }
    case WM_DESTROY:
        PostQuitMessage(0);
        break;
    }
    return DefWindowProcW(hwnd, uMsg, wParam, lParam);
}


// The main entry point for a Windows GUI application
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    const wchar_t CLASS_NAME[] = L"SusiCompilerAppClass";

    WNDCLASSW wc = { }; // Use WNDCLASSW
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;


    RegisterClassW(&wc); // Use RegisterClassW


    // Create the main window
    HWND hwnd = CreateWindowExW(
        0, CLASS_NAME, L"Serio V1.0 Compiler", WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT, 800, 600,
        NULL, NULL, hInstance, NULL
    );


    if (hwnd == NULL) {
        return 0;
    }


    ShowWindow(hwnd, nCmdShow);
    UpdateWindow(hwnd);


    // Message loop
    MSG msg;
    while (GetMessageW(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }


    return 0;
}
