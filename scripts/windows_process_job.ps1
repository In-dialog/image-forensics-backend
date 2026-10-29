# Use in a dedicated launcher PowerShell process, before starting any services.
# The OS closes this process's non-inherited job handle even when the terminal
# is closed or the launcher is forcibly terminated and finally cannot execute.
function Enable-LauncherProcessCleanup {
    if (-not ('ImageForensics.LauncherLifetime' -as [type])) {
        Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

namespace ImageForensics {
    public static class LauncherLifetime {
        private static IntPtr job = IntPtr.Zero;

        [StructLayout(LayoutKind.Sequential)]
        private struct BasicLimits {
            public long ProcessTime, JobTime;
            public uint Flags;
            public UIntPtr MinimumWorkingSet, MaximumWorkingSet;
            public uint ActiveProcesses;
            public UIntPtr Affinity;
            public uint Priority, Scheduling;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct IoCounters {
            public ulong ReadOperations, WriteOperations, OtherOperations;
            public ulong ReadBytes, WriteBytes, OtherBytes;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct ExtendedLimits {
            public BasicLimits Basic;
            public IoCounters Io;
            public UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory;
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr CreateJobObject(IntPtr attributes, string name);
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool SetInformationJobObject(IntPtr job, int infoClass,
            ref ExtendedLimits limits, uint size);
        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
        [DllImport("kernel32.dll")]
        private static extern IntPtr GetCurrentProcess();
        [DllImport("kernel32.dll")]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool CloseHandle(IntPtr handle);

        public static void Attach() {
            if (job != IntPtr.Zero) return;
            // A null security descriptor gives a non-inheritable handle. Child
            // processes inherit membership, but cannot keep the job alive.
            IntPtr created = CreateJobObject(IntPtr.Zero, null);
            if (created == IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error());
            var limits = new ExtendedLimits();
            limits.Basic.Flags = 0x2000; // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if (!SetInformationJobObject(created, 9, ref limits,
                    (uint)Marshal.SizeOf(typeof(ExtendedLimits))) ||
                !AssignProcessToJobObject(created, GetCurrentProcess())) {
                int error = Marshal.GetLastWin32Error();
                CloseHandle(created);
                throw new Win32Exception(error, "Could not protect launcher process cleanup");
            }
            // Kept until this dedicated launcher process exits. No breakaway
            // flags: Python, Node and their descendants stay in the same job.
            job = created;
        }
    }
}
'@
    }
    [ImageForensics.LauncherLifetime]::Attach()
}
